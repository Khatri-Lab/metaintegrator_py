"""
UCSC Database Connectors for Gene Annotation.
Production-grade implementation of UCSC GenBank and RefSeq database connectors.

This module provides robust connections to UCSC Genome Browser databases
for mapping GenBank and RefSeq accession numbers to gene symbols.
"""

import sqlite3
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError

    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    warnings.warn("mysql-connector-python not available. UCSC database features will be limited.")


@dataclass
class UCSDConnectionConfig:
    """Configuration for UCSC database connections."""

    host: str = "genome-mysql.cse.ucsc.edu"
    username: str = "genome"
    password: str = ""
    port: int = 3306
    connection_timeout: int = 30
    read_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0


class UCSDConnectionPool:
    """
    Connection pool manager for UCSC database connections.
    Handles connection reuse and automatic cleanup.
    """

    def __init__(self, config: UCSDConnectionConfig, max_connections: int = 5) -> None:
        """Initialize connection pool."""
        self.config = config
        self.max_connections = max_connections
        self._connections = {}
        self._connection_times = {}
        self._max_connection_age = timedelta(minutes=10)  # Refresh connections after 10 min

    @contextmanager
    def get_connection(self, database: str) -> Iterator[Any]:
        """Get a database connection from the pool."""
        if not MYSQL_AVAILABLE:
            raise RuntimeError("MySQL connector not available")

        conn = None
        try:
            conn = self._get_or_create_connection(database)
            if conn and conn.is_connected():
                yield conn
            else:
                raise MySQLError("Failed to establish connection")
        except Exception as e:
            if conn:
                self._cleanup_connection(database)
            raise e

    def _get_or_create_connection(self, database: str) -> Any:
        """Get existing connection or create new one."""
        now = datetime.now()

        # Check if we have a valid existing connection
        if database in self._connections:
            conn = self._connections[database]
            conn_time = self._connection_times.get(database, now)

            # Check if connection is still valid and not too old
            if conn and conn.is_connected() and now - conn_time < self._max_connection_age:
                return conn
            else:
                self._cleanup_connection(database)

        # Create new connection
        try:
            conn = mysql.connector.connect(
                host=self.config.host,
                user=self.config.username,
                password=self.config.password,
                database=database,
                port=self.config.port,
                connection_timeout=self.config.connection_timeout,
                autocommit=True,
                sql_mode="TRADITIONAL",
                use_unicode=True,
                charset="utf8",
            )

            self._connections[database] = conn
            self._connection_times[database] = now
            return conn

        except MySQLError as e:
            raise MySQLError(f"Failed to connect to UCSC database {database}: {str(e)}")

    def _cleanup_connection(self, database: str) -> None:
        """Clean up a specific database connection."""
        if database in self._connections:
            try:
                conn = self._connections[database]
                if conn and conn.is_connected():
                    conn.close()
            except:
                pass  # Ignore cleanup errors
            finally:
                del self._connections[database]
                if database in self._connection_times:
                    del self._connection_times[database]

    def cleanup_all(self) -> None:
        """Clean up all connections."""
        for database in list(self._connections.keys()):
            self._cleanup_connection(database)


class UCSDGenBankConnector:
    """
    Production-grade connector for UCSC GenBank annotation data.

    Maps GenBank accession numbers to gene symbols using the UCSC hgFixed database.
    Includes caching, error handling, and connection pooling.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        config: Optional[UCSDConnectionConfig] = None,
    ) -> None:
        """
        Initialize GenBank connector.

        Args:
            cache_dir: Directory for caching annotation tables
            config: Database connection configuration
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".ucsc_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or UCSDConnectionConfig()
        self.connection_pool = UCSDConnectionPool(self.config)

        # Cache settings
        self.cache_expiry = timedelta(days=7)  # Cache for 1 week
        self.cache_file = self.cache_dir / "ucsc_genbank_table.pkl"

        # Initialize local cache database
        self.local_db = self.cache_dir / "ucsc_genbank.db"
        self._init_local_cache()

    def _init_local_cache(self) -> None:
        """Initialize local SQLite cache."""
        try:
            with sqlite3.connect(self.local_db) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                               CREATE TABLE IF NOT EXISTS genbank_mappings
                               (
                                   acc
                                   TEXT
                                   PRIMARY
                                   KEY,
                                   gene_name
                                   TEXT,
                                   organism
                                   INTEGER,
                                   last_updated
                                   TIMESTAMP
                                   DEFAULT
                                   CURRENT_TIMESTAMP
                               )
                               """)

                cursor.execute("""
                               CREATE INDEX IF NOT EXISTS idx_acc ON genbank_mappings(acc)
                               """)

                cursor.execute("""
                               CREATE INDEX IF NOT EXISTS idx_gene ON genbank_mappings(gene_name)
                               """)

                conn.commit()
        except sqlite3.Error as e:
            warnings.warn(f"Failed to initialize local cache: {str(e)}")

    def get_genbank_table(self, use_cache: bool = True, organism: int = 448) -> pd.DataFrame:
        """
        Get complete GenBank mapping table.

        Args:
            use_cache: Whether to use cached data
            organism: UCSC organism code (448 = human)

        Returns:
            DataFrame with columns: ['name', 'acc'] (gene_name, genbank_accession)
        """
        # Try cache first if enabled
        if use_cache:
            cached_data = self._load_from_cache(organism)
            if cached_data is not None:
                return cached_data

        # Fetch from remote database
        try:
            remote_data = self._fetch_genbank_table_remote(organism)
            if remote_data is not None and not remote_data.empty:
                # Cache the result
                self._save_to_cache(remote_data, organism)
                return remote_data
        except Exception as e:
            warnings.warn(f"Failed to fetch from remote UCSC database: {str(e)}")

        # Fallback to any cached data, even if expired
        fallback_data = self._load_from_cache(organism, ignore_expiry=True)
        if fallback_data is not None:
            warnings.warn("Using expired cache data due to remote fetch failure")
            return fallback_data

        # Return empty DataFrame if all else fails
        warnings.warn("No GenBank data available - returning empty table")
        return pd.DataFrame(columns=["name", "acc"])

    def _fetch_genbank_table_remote(self, organism: int) -> Optional[pd.DataFrame]:
        """Fetch GenBank table from remote UCSC database."""
        database = "hgFixed"
        query = """
                SELECT geneName.name, gbCdnaInfo.acc
                FROM gbCdnaInfo
                         INNER JOIN geneName ON geneName.id = gbCdnaInfo.geneName
                WHERE gbCdnaInfo.organism = %s
                  AND geneName.name != 'n/a' 
            AND geneName.name != 'unknown'
            AND geneName.name IS NOT NULL
            AND gbCdnaInfo.acc IS NOT NULL \
                """

        for attempt in range(self.config.max_retries):
            try:
                with self.connection_pool.get_connection(database) as conn:
                    # Execute query with proper parameter binding
                    cursor = conn.cursor(buffered=True)
                    cursor.execute(query, (organism,))

                    # Fetch results
                    results = cursor.fetchall()
                    cursor.close()

                    # Convert to DataFrame
                    if results:
                        df = pd.DataFrame(results, columns=["name", "acc"])

                        # Clean and validate data
                        df = df.dropna().drop_duplicates()
                        df = df[df["name"].str.strip() != ""]
                        df = df[df["acc"].str.strip() != ""]

                        print(f"Successfully fetched {len(df)} GenBank mappings from UCSC")
                        return df
                    else:
                        warnings.warn("No data returned from UCSC GenBank query")
                        return pd.DataFrame(columns=["name", "acc"])

            except MySQLError as e:
                if attempt == self.config.max_retries - 1:
                    raise MySQLError(
                        f"Failed to fetch GenBank table after {self.config.max_retries} attempts: {str(e)}"
                    )

                warnings.warn(f"Attempt {attempt + 1} failed, retrying: {str(e)}")
                time.sleep(self.config.retry_delay * (attempt + 1))

        return None

    def _load_from_cache(
        self, organism: int, ignore_expiry: bool = False
    ) -> Optional[pd.DataFrame]:
        """Load data from local cache."""
        try:
            with sqlite3.connect(self.local_db) as conn:
                # Check if we have recent data
                if not ignore_expiry:
                    expiry_date = datetime.now() - self.cache_expiry
                    query = """
                            SELECT COUNT(*) \
                            FROM genbank_mappings
                            WHERE organism = ? \
                              AND last_updated > ? \
                            """
                    cursor = conn.cursor()
                    cursor.execute(query, (organism, expiry_date.isoformat()))
                    count = cursor.fetchone()[0]

                    if count == 0:
                        return None

                # Load cached data
                query = """
                        SELECT gene_name as name, acc
                        FROM genbank_mappings
                        WHERE organism = ? \
                        """
                df = pd.read_sql_query(query, conn, params=(organism,))

                if not df.empty:
                    print(f"Loaded {len(df)} GenBank mappings from cache")
                    return df

        except (sqlite3.Error, Exception) as e:
            warnings.warn(f"Failed to load from cache: {str(e)}")

        return None

    def _save_to_cache(self, df: pd.DataFrame, organism: int) -> None:
        """Save data to local cache."""
        try:
            with sqlite3.connect(self.local_db) as conn:
                # Clear old data for this organism
                cursor = conn.cursor()
                cursor.execute("DELETE FROM genbank_mappings WHERE organism = ?", (organism,))

                # Insert new data
                for _, row in df.iterrows():
                    cursor.execute(
                        "INSERT INTO genbank_mappings (acc, gene_name, organism) VALUES (?, ?, ?)",
                        (row["acc"], row["name"], organism),
                    )

                conn.commit()
                print(f"Cached {len(df)} GenBank mappings")

        except sqlite3.Error as e:
            warnings.warn(f"Failed to save to cache: {str(e)}")

    def map_genbank_to_genes(self, genbank_ids: List[str], organism: int = 448) -> Dict[str, str]:
        """
        Map GenBank accession numbers to gene symbols.

        Args:
            genbank_ids: List of GenBank accession numbers
            organism: UCSC organism code

        Returns:
            Dictionary mapping GenBank IDs to gene symbols
        """
        if not genbank_ids:
            return {}

        # Clean input IDs (remove version numbers)
        clean_ids = []
        id_mapping = {}  # Maps clean ID back to original

        for genbank_id in genbank_ids:
            if pd.isna(genbank_id):
                continue

            clean_id = str(genbank_id).strip()
            # Remove version number (everything after last dot)
            if "." in clean_id:
                clean_id = clean_id.rsplit(".", 1)[0]

            clean_ids.append(clean_id)
            id_mapping[clean_id] = genbank_id

        if not clean_ids:
            return {}

        # Get mapping table
        genbank_table = self.get_genbank_table(organism=organism)

        if genbank_table.empty:
            return {}

        # Create mapping
        result_mapping = {}

        for clean_id in clean_ids:
            original_id = id_mapping[clean_id]

            # Look for exact match first
            matches = genbank_table[genbank_table["acc"] == clean_id]

            if matches.empty:
                # Try partial match (in case of versioning issues)
                matches = genbank_table[genbank_table["acc"].str.startswith(clean_id)]

            if not matches.empty:
                # Take the first match
                gene_name = matches.iloc[0]["name"]
                result_mapping[original_id] = gene_name

        return result_mapping

    def cleanup(self) -> None:
        """Clean up resources."""
        self.connection_pool.cleanup_all()


class UCSDRefSeqConnector:
    """
    Production-grade connector for UCSC RefSeq annotation data.

    Maps RefSeq accession numbers to gene symbols using the UCSC hg19 database.
    Includes caching, error handling, and connection pooling.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        config: Optional[UCSDConnectionConfig] = None,
    ) -> None:
        """
        Initialize RefSeq connector.

        Args:
            cache_dir: Directory for caching annotation tables
            config: Database connection configuration
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".ucsc_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or UCSDConnectionConfig()
        self.connection_pool = UCSDConnectionPool(self.config)

        # Cache settings
        self.cache_expiry = timedelta(days=7)  # Cache for 1 week

        # Initialize local cache database
        self.local_db = self.cache_dir / "ucsc_refseq.db"
        self._init_local_cache()

    def _init_local_cache(self) -> None:
        """Initialize local SQLite cache."""
        try:
            with sqlite3.connect(self.local_db) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                               CREATE TABLE IF NOT EXISTS refseq_mappings
                               (
                                   name
                                   TEXT
                                   PRIMARY
                                   KEY,
                                   name2
                                   TEXT,
                                   last_updated
                                   TIMESTAMP
                                   DEFAULT
                                   CURRENT_TIMESTAMP
                               )
                               """)

                cursor.execute("""
                               CREATE INDEX IF NOT EXISTS idx_name ON refseq_mappings(name)
                               """)

                cursor.execute("""
                               CREATE INDEX IF NOT EXISTS idx_name2 ON refseq_mappings(name2)
                               """)

                conn.commit()
        except sqlite3.Error as e:
            warnings.warn(f"Failed to initialize local cache: {str(e)}")

    def get_refseq_table(self, use_cache: bool = True, database: str = "hg19") -> pd.DataFrame:
        """
        Get complete RefSeq mapping table.

        Args:
            use_cache: Whether to use cached data
            database: UCSC database name (hg19, hg38, etc.)

        Returns:
            DataFrame with columns: ['name', 'name2'] (refseq_id, gene_symbol)
        """
        # Try cache first if enabled
        if use_cache:
            cached_data = self._load_from_cache(database)
            if cached_data is not None:
                return cached_data

        # Fetch from remote database
        try:
            remote_data = self._fetch_refseq_table_remote(database)
            if remote_data is not None and not remote_data.empty:
                # Cache the result
                self._save_to_cache(remote_data, database)
                return remote_data
        except Exception as e:
            warnings.warn(f"Failed to fetch from remote UCSC database: {str(e)}")

        # Fallback to any cached data, even if expired
        fallback_data = self._load_from_cache(database, ignore_expiry=True)
        if fallback_data is not None:
            warnings.warn("Using expired cache data due to remote fetch failure")
            return fallback_data

        # Return empty DataFrame if all else fails
        warnings.warn("No RefSeq data available - returning empty table")
        return pd.DataFrame(columns=["name", "name2"])

    def _fetch_refseq_table_remote(self, database: str) -> Optional[pd.DataFrame]:
        """Fetch RefSeq table from remote UCSC database."""
        query = """
                SELECT name, name2
                FROM refGene
                WHERE name IS NOT NULL
                  AND name2 IS NOT NULL
                  AND name2 != '' \
                """

        for attempt in range(self.config.max_retries):
            try:
                with self.connection_pool.get_connection(database) as conn:
                    # Execute query
                    cursor = conn.cursor(buffered=True)
                    cursor.execute(query)

                    # Fetch results
                    results = cursor.fetchall()
                    cursor.close()

                    # Convert to DataFrame
                    if results:
                        df = pd.DataFrame(results, columns=["name", "name2"])

                        # Clean and validate data
                        df = df.dropna().drop_duplicates()
                        df = df[df["name"].str.strip() != ""]
                        df = df[df["name2"].str.strip() != ""]

                        print(f"Successfully fetched {len(df)} RefSeq mappings from UCSC")
                        return df
                    else:
                        warnings.warn("No data returned from UCSC RefSeq query")
                        return pd.DataFrame(columns=["name", "name2"])

            except MySQLError as e:
                if attempt == self.config.max_retries - 1:
                    raise MySQLError(
                        f"Failed to fetch RefSeq table after {self.config.max_retries} attempts: {str(e)}"
                    )

                warnings.warn(f"Attempt {attempt + 1} failed, retrying: {str(e)}")
                time.sleep(self.config.retry_delay * (attempt + 1))

        return None

    def _load_from_cache(
        self, database: str, ignore_expiry: bool = False
    ) -> Optional[pd.DataFrame]:
        """Load data from local cache."""
        try:
            with sqlite3.connect(self.local_db) as conn:
                # Check if we have recent data
                if not ignore_expiry:
                    expiry_date = datetime.now() - self.cache_expiry
                    query = """
                            SELECT COUNT(*) \
                            FROM refseq_mappings
                            WHERE last_updated > ? \
                            """
                    cursor = conn.cursor()
                    cursor.execute(query, (expiry_date.isoformat(),))
                    count = cursor.fetchone()[0]

                    if count == 0:
                        return None

                # Load cached data
                query = "SELECT name, name2 FROM refseq_mappings"
                df = pd.read_sql_query(query, conn)

                if not df.empty:
                    print(f"Loaded {len(df)} RefSeq mappings from cache")
                    return df

        except (sqlite3.Error, Exception) as e:
            warnings.warn(f"Failed to load from cache: {str(e)}")

        return None

    def _save_to_cache(self, df: pd.DataFrame, database: str) -> None:
        """Save data to local cache."""
        try:
            with sqlite3.connect(self.local_db) as conn:
                # Clear old data
                cursor = conn.cursor()
                cursor.execute("DELETE FROM refseq_mappings")

                # Insert new data
                for _, row in df.iterrows():
                    cursor.execute(
                        "INSERT INTO refseq_mappings (name, name2) VALUES (?, ?)",
                        (row["name"], row["name2"]),
                    )

                conn.commit()
                print(f"Cached {len(df)} RefSeq mappings")

        except sqlite3.Error as e:
            warnings.warn(f"Failed to save to cache: {str(e)}")

    def map_refseq_to_genes(self, refseq_ids: List[str], database: str = "hg19") -> Dict[str, str]:
        """
        Map RefSeq accession numbers to gene symbols.

        Args:
            refseq_ids: List of RefSeq accession numbers
            database: UCSC database name

        Returns:
            Dictionary mapping RefSeq IDs to gene symbols
        """
        if not refseq_ids:
            return {}

        # Clean input IDs (remove version numbers and _at suffixes)
        clean_ids = []
        id_mapping = {}  # Maps clean ID back to original

        for refseq_id in refseq_ids:
            if pd.isna(refseq_id):
                continue

            clean_id = str(refseq_id).strip()

            # Remove version number (everything after last dot)
            if "." in clean_id:
                clean_id = clean_id.rsplit(".", 1)[0]

            # Remove _at suffix if present (Affymetrix probe set naming)
            if clean_id.endswith("_at"):
                clean_id = clean_id[:-3]

            clean_ids.append(clean_id)
            id_mapping[clean_id] = refseq_id

        if not clean_ids:
            return {}

        # Get mapping table
        refseq_table = self.get_refseq_table(database=database)

        if refseq_table.empty:
            return {}

        # Create mapping
        result_mapping = {}

        for clean_id in clean_ids:
            original_id = id_mapping[clean_id]

            # Look for exact match first
            matches = refseq_table[refseq_table["name"] == clean_id]

            if matches.empty:
                # Try partial match (in case of versioning issues)
                matches = refseq_table[refseq_table["name"].str.startswith(clean_id)]

            if not matches.empty:
                # Take the first match
                gene_name = matches.iloc[0]["name2"]
                result_mapping[original_id] = gene_name

        return result_mapping

    def cleanup(self) -> None:
        """Clean up resources."""
        self.connection_pool.cleanup_all()


# Convenience functions
def get_ucsc_genbank_table(cache_dir: Optional[str] = None, organism: int = 448) -> pd.DataFrame:
    """
    Convenience function to get UCSC GenBank mapping table.

    Args:
        cache_dir: Cache directory
        organism: UCSC organism code (448 = human)

    Returns:
        DataFrame with GenBank to gene symbol mappings
    """
    connector = UCSDGenBankConnector(cache_dir=cache_dir)
    try:
        return connector.get_genbank_table(organism=organism)
    finally:
        connector.cleanup()


def get_ucsc_refseq_table(cache_dir: Optional[str] = None, database: str = "hg19") -> pd.DataFrame:
    """
    Convenience function to get UCSC RefSeq mapping table.

    Args:
        cache_dir: Cache directory
        database: UCSC database name

    Returns:
        DataFrame with RefSeq to gene symbol mappings
    """
    connector = UCSDRefSeqConnector(cache_dir=cache_dir)
    try:
        return connector.get_refseq_table(database=database)
    finally:
        connector.cleanup()


def map_genbank_ids(
    genbank_ids: List[str], cache_dir: Optional[str] = None, organism: int = 448
) -> Dict[str, str]:
    """
    Convenience function to map GenBank IDs to gene symbols.

    Args:
        genbank_ids: List of GenBank accession numbers
        cache_dir: Cache directory
        organism: UCSC organism code

    Returns:
        Dictionary mapping GenBank IDs to gene symbols
    """
    connector = UCSDGenBankConnector(cache_dir=cache_dir)
    try:
        return connector.map_genbank_to_genes(genbank_ids, organism=organism)
    finally:
        connector.cleanup()


def map_refseq_ids(
    refseq_ids: List[str], cache_dir: Optional[str] = None, database: str = "hg19"
) -> Dict[str, str]:
    """
    Convenience function to map RefSeq IDs to gene symbols.

    Args:
        refseq_ids: List of RefSeq accession numbers
        cache_dir: Cache directory
        database: UCSC database name

    Returns:
        Dictionary mapping RefSeq IDs to gene symbols
    """
    connector = UCSDRefSeqConnector(cache_dir=cache_dir)
    try:
        return connector.map_refseq_to_genes(refseq_ids, database=database)
    finally:
        connector.cleanup()
