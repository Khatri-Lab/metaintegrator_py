"""
Comprehensive unit tests for UCSC Database Connectors.
Tests connection handling, caching, ID mapping, and error scenarios.
"""

import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
import warnings
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# Import the modules we're testing
from metaintegrator.ucsc_connectors import (
    UCSDConnectionConfig,
    UCSDConnectionPool,
    UCSDGenBankConnector,
    UCSDRefSeqConnector,
    get_ucsc_genbank_table,
    get_ucsc_refseq_table,
    map_genbank_ids,
    map_refseq_ids,
)


class TestUCSDConnectionConfig(unittest.TestCase):
    """Test UCSD connection configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UCSDConnectionConfig()

        self.assertEqual(config.host, "genome-mysql.cse.ucsc.edu")
        self.assertEqual(config.username, "genome")
        self.assertEqual(config.password, "")
        self.assertEqual(config.port, 3306)
        self.assertEqual(config.connection_timeout, 30)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.retry_delay, 1.0)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = UCSDConnectionConfig(
            host="custom.host.com",
            username="testuser",
            password="testpass",
            port=3307,
            max_retries=5,
        )

        self.assertEqual(config.host, "custom.host.com")
        self.assertEqual(config.username, "testuser")
        self.assertEqual(config.password, "testpass")
        self.assertEqual(config.port, 3307)
        self.assertEqual(config.max_retries, 5)


class TestUCSDConnectionPool(unittest.TestCase):
    """Test UCSD connection pool functionality."""

    def setUp(self):
        """Set up test environment."""
        self.config = UCSDConnectionConfig()
        self.pool = UCSDConnectionPool(self.config, max_connections=2)

    def tearDown(self):
        """Clean up after tests."""
        self.pool.cleanup_all()

    @patch("metaintegrator.ucsc_connectors.MYSQL_AVAILABLE", True)
    @patch("metaintegrator.ucsc_connectors.mysql.connector.connect")
    def test_successful_connection(self, mock_connect):
        """Test successful database connection."""
        # Mock successful connection
        mock_conn = Mock()
        mock_conn.is_connected.return_value = True
        mock_connect.return_value = mock_conn

        with self.pool.get_connection("test_db") as conn:
            self.assertIsNotNone(conn)
            self.assertTrue(conn.is_connected())

        # Verify connection parameters
        mock_connect.assert_called_once()
        call_args = mock_connect.call_args[1]
        self.assertEqual(call_args["host"], self.config.host)
        self.assertEqual(call_args["user"], self.config.username)
        self.assertEqual(call_args["database"], "test_db")

    @patch("metaintegrator.ucsc_connectors.MYSQL_AVAILABLE", False)
    def test_mysql_unavailable(self):
        """Test behavior when MySQL is not available."""
        with self.assertRaises(RuntimeError):
            with self.pool.get_connection("test_db"):
                pass

    @patch("metaintegrator.ucsc_connectors.MYSQL_AVAILABLE", True)
    @patch("metaintegrator.ucsc_connectors.mysql.connector.connect")
    def test_connection_failure(self, mock_connect):
        """Test handling of connection failures."""
        from metaintegrator.ucsc_connectors import MySQLError

        # Mock connection failure
        mock_connect.side_effect = MySQLError("Connection failed")

        with self.assertRaises(MySQLError):
            with self.pool.get_connection("test_db"):
                pass

    @patch("metaintegrator.ucsc_connectors.MYSQL_AVAILABLE", True)
    @patch("metaintegrator.ucsc_connectors.mysql.connector.connect")
    def test_connection_reuse(self, mock_connect):
        """Test connection reuse in pool."""
        # Mock successful connection
        mock_conn = Mock()
        mock_conn.is_connected.return_value = True
        mock_connect.return_value = mock_conn

        # First connection
        with self.pool.get_connection("test_db"):
            pass

        # Second connection should reuse
        with self.pool.get_connection("test_db"):
            pass

        # Should only create one connection
        mock_connect.assert_called_once()


class TestUCSDGenBankConnector(unittest.TestCase):
    """Test UCSD GenBank connector functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.connector = UCSDGenBankConnector(cache_dir=self.temp_dir)

    def tearDown(self):
        """Clean up after tests."""
        self.connector.cleanup()
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test connector initialization."""
        self.assertTrue(self.connector.cache_dir.exists())
        self.assertTrue(self.connector.local_db.exists())

        # Verify database schema
        with sqlite3.connect(self.connector.local_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='genbank_mappings'"
            )
            self.assertIsNotNone(cursor.fetchone())

    def test_cache_save_and_load(self):
        """Test caching functionality."""
        # Create test data
        test_data = pd.DataFrame(
            {"name": ["GENE1", "GENE2", "GENE3"], "acc": ["ACC1", "ACC2", "ACC3"]}
        )

        # Save to cache
        self.connector._save_to_cache(test_data, organism=448)

        # Load from cache
        loaded_data = self.connector._load_from_cache(organism=448)

        self.assertIsNotNone(loaded_data)
        self.assertEqual(len(loaded_data), 3)
        pd.testing.assert_frame_equal(loaded_data, test_data)

    def test_cache_expiry(self):
        """Test cache expiry functionality."""
        # Create test data
        test_data = pd.DataFrame({"name": ["GENE1"], "acc": ["ACC1"]})

        # Save to cache
        self.connector._save_to_cache(test_data, organism=448)

        # Manually update timestamp to be old
        with sqlite3.connect(self.connector.local_db) as conn:
            cursor = conn.cursor()
            old_date = datetime.now() - timedelta(days=10)
            cursor.execute("UPDATE genbank_mappings SET last_updated = ?", (old_date.isoformat(),))
            conn.commit()

        # Should return None for expired data
        self.assertIsNone(self.connector._load_from_cache(organism=448))

        # But should return data if expiry is ignored
        expired_data = self.connector._load_from_cache(organism=448, ignore_expiry=True)
        self.assertIsNotNone(expired_data)

    @patch.object(UCSDGenBankConnector, "_fetch_genbank_table_remote")
    def test_get_genbank_table_remote_success(self, mock_fetch):
        """Test successful remote fetch."""
        # Mock remote fetch success
        test_data = pd.DataFrame({"name": ["GAPDH", "ACTB"], "acc": ["M33197", "NM_001101"]})
        mock_fetch.return_value = test_data

        result = self.connector.get_genbank_table(use_cache=False)

        pd.testing.assert_frame_equal(result, test_data)
        mock_fetch.assert_called_once_with(448)

    @patch.object(UCSDGenBankConnector, "_fetch_genbank_table_remote")
    @patch.object(UCSDGenBankConnector, "_load_from_cache")
    def test_get_genbank_table_fallback_to_cache(self, mock_load, mock_fetch):
        """Test fallback to cache when remote fetch fails."""
        # Mock remote fetch failure
        mock_fetch.side_effect = Exception("Connection failed")

        # Mock cache data
        cache_data = pd.DataFrame({"name": ["GENE1"], "acc": ["ACC1"]})
        mock_load.return_value = cache_data

        # Force the warnings filter to record every warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = self.connector.get_genbank_table(use_cache=False)

            # Should use cache data
            pd.testing.assert_frame_equal(result, cache_data)

            print(f"\n::: test_get_genbank_table_fallback_to_cache() w={w}")
            # Should issue warning
            self.assertTrue(any("expired cache data" in str(warning.message) for warning in w))

    def test_map_genbank_to_genes(self):
        """Test GenBank ID to gene symbol mapping."""
        # Set up test data in cache
        test_data = pd.DataFrame(
            {
                "name": ["GAPDH", "ACTB", "TP53"],
                "acc": ["M33197", "NM_001101", "M13994"],
            }
        )
        self.connector._save_to_cache(test_data, organism=448)

        # Test mapping
        genbank_ids = ["M33197", "NM_001101.2", "UNKNOWN"]
        result = self.connector.map_genbank_to_genes(genbank_ids)

        expected = {
            "M33197": "GAPDH",
            "NM_001101.2": "ACTB",  # Version number should be stripped
        }

        self.assertEqual(result, expected)

    def test_map_genbank_empty_input(self):
        """Test mapping with empty input."""
        result = self.connector.map_genbank_to_genes([])
        self.assertEqual(result, {})

        result = self.connector.map_genbank_to_genes([None, "", "   "])
        self.assertEqual(result, {})

    @patch("metaintegrator.ucsc_connectors.MYSQL_AVAILABLE", True)
    @patch.object(UCSDConnectionPool, "get_connection")
    def test_fetch_genbank_table_remote(self, mock_get_conn):
        """Test remote GenBank table fetch."""
        # Mock database connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()

        # Mock query results
        mock_cursor.fetchall.return_value = [("GAPDH", "M33197"), ("ACTB", "NM_001101")]

        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        result = self.connector._fetch_genbank_table_remote(organism=448)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result.columns), ["name", "acc"])

    @patch("metaintegrator.ucsc_connectors.MYSQL_AVAILABLE", True)
    @patch.object(UCSDConnectionPool, "get_connection")
    def test_fetch_genbank_table_retry(self, mock_get_conn):
        """Test retry logic for remote fetch."""
        from metaintegrator.ucsc_connectors import MySQLError

        # Mock connection to fail twice, then succeed
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [("GAPDH", "M33197")]
        mock_conn.cursor.return_value = mock_cursor

        mock_get_conn.side_effect = [
            MySQLError("Connection failed"),
            MySQLError("Connection failed"),
            Mock(__enter__=Mock(return_value=mock_conn), __exit__=Mock(return_value=None)),
        ]

        # Should succeed after retries
        result = self.connector._fetch_genbank_table_remote(organism=448)
        self.assertIsNotNone(result)

        # Verify retry attempts
        self.assertEqual(mock_get_conn.call_count, 3)


class TestUCSDRefSeqConnector(unittest.TestCase):
    """Test UCSD RefSeq connector functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.connector = UCSDRefSeqConnector(cache_dir=self.temp_dir)

    def tearDown(self):
        """Clean up after tests."""
        self.connector.cleanup()
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test connector initialization."""
        self.assertTrue(self.connector.cache_dir.exists())
        self.assertTrue(self.connector.local_db.exists())

        # Verify database schema
        with sqlite3.connect(self.connector.local_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='refseq_mappings'"
            )
            self.assertIsNotNone(cursor.fetchone())

    def test_cache_functionality(self):
        """Test RefSeq caching functionality."""
        # Create test data
        test_data = pd.DataFrame(
            {
                "name": ["NM_001101", "NM_000546", "NM_002046"],
                "name2": ["ACTB", "TP53", "GAPDH"],
            }
        )

        # Save to cache
        self.connector._save_to_cache(test_data, database="hg19")

        # Load from cache
        loaded_data = self.connector._load_from_cache(database="hg19")

        self.assertIsNotNone(loaded_data)
        pd.testing.assert_frame_equal(loaded_data, test_data)

    def test_map_refseq_to_genes(self):
        """Test RefSeq ID to gene symbol mapping."""
        # Set up test data in cache
        test_data = pd.DataFrame(
            {
                "name": ["NM_001101", "NM_000546", "NM_002046"],
                "name2": ["ACTB", "TP53", "GAPDH"],
            }
        )
        self.connector._save_to_cache(test_data, database="hg19")

        # Test mapping with various formats
        refseq_ids = ["NM_001101", "NM_000546.5", "NM_002046_at", "UNKNOWN"]
        result = self.connector.map_refseq_to_genes(refseq_ids)

        expected = {"NM_001101": "ACTB", "NM_000546.5": "TP53", "NM_002046_at": "GAPDH"}

        self.assertEqual(result, expected)

    @patch("metaintegrator.ucsc_connectors.MYSQL_AVAILABLE", True)
    @patch.object(UCSDConnectionPool, "get_connection")
    def test_fetch_refseq_table_remote(self, mock_get_conn):
        """Test remote RefSeq table fetch."""
        # Mock database connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()

        # Mock query results
        mock_cursor.fetchall.return_value = [
            ("NM_001101", "ACTB"),
            ("NM_000546", "TP53"),
        ]

        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        result = self.connector._fetch_refseq_table_remote(database="hg19")

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result.columns), ["name", "name2"])

    @patch.object(UCSDRefSeqConnector, "_fetch_refseq_table_remote")
    def test_get_refseq_table_empty_response(self, mock_fetch):
        """Test handling of empty remote response."""
        mock_fetch.return_value = pd.DataFrame(columns=["name", "name2"])

        result = self.connector.get_refseq_table(use_cache=False)

        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ["name", "name2"])


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    @patch.object(UCSDGenBankConnector, "get_genbank_table")
    def test_get_ucsc_genbank_table(self, mock_get_table):
        """Test convenience function for getting GenBank table."""
        test_data = pd.DataFrame({"name": ["GENE1"], "acc": ["ACC1"]})
        mock_get_table.return_value = test_data

        result = get_ucsc_genbank_table(cache_dir=self.temp_dir, organism=448)

        pd.testing.assert_frame_equal(result, test_data)
        mock_get_table.assert_called_once_with(organism=448)

    @patch.object(UCSDRefSeqConnector, "get_refseq_table")
    def test_get_ucsc_refseq_table(self, mock_get_table):
        """Test convenience function for getting RefSeq table."""
        test_data = pd.DataFrame({"name": ["REF1"], "name2": ["GENE1"]})
        mock_get_table.return_value = test_data

        result = get_ucsc_refseq_table(cache_dir=self.temp_dir, database="hg19")

        pd.testing.assert_frame_equal(result, test_data)
        mock_get_table.assert_called_once_with(database="hg19")

    @patch.object(UCSDGenBankConnector, "map_genbank_to_genes")
    def test_map_genbank_ids(self, mock_map):
        """Test convenience function for mapping GenBank IDs."""
        mock_map.return_value = {"ACC1": "GENE1"}

        result = map_genbank_ids(["ACC1"], cache_dir=self.temp_dir, organism=448)

        self.assertEqual(result, {"ACC1": "GENE1"})
        mock_map.assert_called_once_with(["ACC1"], organism=448)

    @patch.object(UCSDRefSeqConnector, "map_refseq_to_genes")
    def test_map_refseq_ids(self, mock_map):
        """Test convenience function for mapping RefSeq IDs."""
        mock_map.return_value = {"NM_001": "GENE1"}

        result = map_refseq_ids(["NM_001"], cache_dir=self.temp_dir, database="hg19")

        self.assertEqual(result, {"NM_001": "GENE1"})
        mock_map.assert_called_once_with(["NM_001"], database="hg19")


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    def test_corrupted_cache_database(self):
        """Test handling of corrupted cache database."""
        connector = UCSDGenBankConnector(cache_dir=self.temp_dir)

        # Corrupt the database file
        with open(connector.local_db, "w") as f:
            f.write("corrupted data")

        # Should handle corruption gracefully
        # Force the warnings filter to record every warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This should not raise an exception
            connector._load_from_cache(organism=448)

            print(f"\n::: test_corrupted_cache_database() w={w}")
            # Should issue a warning
            self.assertTrue(len(w) > 0)

    def test_permission_error_cache(self):
        """Test handling of permission errors."""
        # This test would need platform-specific implementation
        # to actually create permission errors
        pass

    @patch("metaintegrator.ucsc_connectors.MYSQL_AVAILABLE", True)
    @patch.object(UCSDConnectionPool, "get_connection")
    def test_database_timeout(self, mock_get_conn):
        """Test handling of database timeout."""
        from metaintegrator.ucsc_connectors import MySQLError

        # Mock timeout error
        mock_get_conn.side_effect = MySQLError("Connection timeout")

        connector = UCSDGenBankConnector(cache_dir=self.temp_dir)

        # Force the warnings filter to record every warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = connector.get_genbank_table(use_cache=False)

            print(f"\n::: test_database_timeout() w={w}")
            # Should return empty DataFrame and issue warning
            self.assertTrue(result.empty)
            self.assertTrue(
                any("No GenBank data available" in str(warning.message) for warning in w)
            )

    def test_invalid_input_data(self):
        """Test handling of invalid input data."""
        connector = UCSDGenBankConnector(cache_dir=self.temp_dir)

        # Test with None and empty inputs
        result = connector.map_genbank_to_genes(None)
        self.assertEqual(result, {})

        result = connector.map_genbank_to_genes([])
        self.assertEqual(result, {})

        # Test with invalid data types
        result = connector.map_genbank_to_genes([None, "", 123, pd.NA])
        self.assertEqual(result, {})


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    @patch.object(UCSDGenBankConnector, "_fetch_genbank_table_remote")
    def test_complete_genbank_workflow(self, mock_fetch):
        """Test complete GenBank annotation workflow."""
        # Mock remote data
        remote_data = pd.DataFrame(
            {
                "name": ["GAPDH", "ACTB", "TP53"],
                "acc": ["M33197", "NM_001101", "M13994"],
            }
        )
        mock_fetch.return_value = remote_data

        # Create connector
        connector = UCSDGenBankConnector(cache_dir=self.temp_dir)

        try:
            # First call should fetch from remote and cache
            table1 = connector.get_genbank_table(use_cache=False)
            pd.testing.assert_frame_equal(table1, remote_data)

            # Second call should use cache
            table2 = connector.get_genbank_table(use_cache=True)
            pd.testing.assert_frame_equal(table2, remote_data)

            # Test mapping
            genbank_ids = ["M33197", "NM_001101.3", "UNKNOWN"]
            mapping = connector.map_genbank_to_genes(genbank_ids)

            expected_mapping = {"M33197": "GAPDH", "NM_001101.3": "ACTB"}
            self.assertEqual(mapping, expected_mapping)

            # Verify only one remote fetch occurred
            mock_fetch.assert_called_once()

        finally:
            connector.cleanup()


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
