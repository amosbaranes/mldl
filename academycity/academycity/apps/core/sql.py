from django.conf import settings
from django.db import connection, models
from django.core.exceptions import FieldDoesNotExist
import psycopg2
import os


class SQL(object):
    def __init__(self):
        # if settings.DEBUG:
        #     self.connection = psycopg2.connect(user = 'academycity', password = 'academycity',
        #                               host = 'localhost', port = 5432, database = 'academycity')
        # else:
        #     self.connection = psycopg2.connect(user=os.environ['DB_USER'], password=os.environ['DB_PASS'],
        #                                        host=os.environ['DB_SERVICE'], port=os.environ['DB_PORT'],
        #                                        database=os.environ['DB_NAME'])
        self.connection = connection

    def exc_sql(self, ssql, data, verb=''):
        try:
            cursor = self.connection.cursor()
            cursor.execute(ssql, data)
            if verb == 'select':
                rows = cursor.fetchall()
                return rows
            else:
                # insert update delete
                self.connection.commit()
                return cursor.rowcount
        except (Exception, psycopg2.Error) as error:
            print(error)
        finally:
            if self.connection:
                cursor.close()
                print("PostgreSQL connection is closed")

    def __del__(self):
        self.connection.close()

    def exc_sql_select(self, ssql, data):
        return self.exc_sql(ssql, data, verb='select')


class TruncateTableMixin(object):

    @classmethod
    def truncate(cls):
        with connection.cursor() as cursor:
            try:
                cursor.execute('TRUNCATE TABLE {} RESTART IDENTITY CASCADE'.format(cls._meta.db_table))
            except Exception as ex:
                # Delete all rows from the table
                cursor.execute(f'DELETE FROM {cls._meta.db_table}')
                # Reset SQLite auto-increment sequence (if it has one)
                cursor.execute("UPDATE sqlite_sequence SET seq = 0 WHERE name = %s", [cls._meta.db_table])

    @classmethod
    def model_field_exists(cls, field):
        try:
            cls._meta.get_field(field)
            return True
        except FieldDoesNotExist:
            return False

