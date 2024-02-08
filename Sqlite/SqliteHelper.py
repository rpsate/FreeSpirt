import sqlite3
import sys

"""
CREATE TABLE vul_database (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ast TEXT,
    ast_vec BLOB,
    ast_features_vec BLOB,
    string_features_vec BLOB,
    block_features_vec BLOB,
    callee_features_vec BLOB
);
"""

class SqliteHelper:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.conn.row_factory = sqlite3.Row
        self.conn.isolation_level = "DEFERRED"
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def execute(self, sql, params=None):
        try:
            if params:
                self.cursor.execute(sql, params)
            else:
                self.cursor.execute(sql)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Error: {e}")
            return False
        return True

    def execute_many(self, sql, params_list):
        try:
            self.cursor.executemany(sql, params_list)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Error: {e}")
            return False
        return True

    def query(self, sql, params=None):
        try:
            if params:
                self.cursor.execute(sql, params)
            else:
                self.cursor.execute(sql)
            rows = self.cursor.fetchall()
            return rows
        except Exception as e:
            print(f"Error: {e}")
            return None

    def insert(self, table_name, data_dict):
        columns = ", ".join(data_dict.keys())
        placeholders = ":" + ", :".join(data_dict.keys())
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        try:
            self.cursor.execute(sql, data_dict)
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error: {e}")
            return False

    def update(self, table_name, data_dict, where_clause="", where_params=None):
        set_clause = ", ".join([f"{key}=:{key}" for key in data_dict.keys()])
        sql = f"UPDATE {table_name} SET {set_clause}"
        if where_clause:
            sql += " WHERE " + where_clause
        params = data_dict.copy()
        if where_params:
            params.update(where_params)
        print(sql)
        print(params)
        try:
            self.cursor.execute(sql, params)
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error: {e}")
            return False

    def delete(self, table_name, where_clause="", where_params=None):
        sql = f"DELETE FROM {table_name}"
        if where_clause:
            sql += " WHERE " + where_clause
        try:
            self.cursor.execute(sql, where_params)
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error: {e}")
            return False

    def close(self):
        self.cursor.close()
        self.conn.close()
