import duckdb
import os

def get_con(path="scania.duckdb", read_only=False):
    return duckdb.connect(path, read_only=read_only)

def build_raw_tables(data_dir="data", db_path="scania.duckdb"):
    con = duckdb.connect(db_path)
    for f in os.listdir(data_dir):
        if f.endswith(".csv"):
            table_name = f.replace(".csv", "")
            con.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT * FROM read_csv_auto('{data_dir}/{f}');
            """)
            count = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
            print(f"  {table_name}: {count:,} rows")
    con.close()
    print(f"\nDatabase saved to {db_path}")

if __name__ == "__main__":
    build_raw_tables()
