import os
import random
from tqdm.auto import tqdm
from faker import Faker
from datetime import datetime, timedelta
import numpy as np
import pandas as pd 

# =========== config ===========
# Large Config
# min_num_records = 200_000
# max_num_record = 900_000
# min_columns = 30
# max_columns = 50
# min_pk_columns = 1
# max_pk_columns = 1
# folder_path = 'large_dataset'
# num_tables =  3

# normal Config
min_num_records = 1_000_000
max_num_record =  1_000_001
min_columns = 3
max_columns = 5
min_pk_columns = 1
max_pk_columns = 1
folder_path = 'data'
num_tables=  1

max_fk_columns = 2
seed = 42
# Initialize Faker instance
fake = Faker()

# Column types and their possible names
column_types = ['int', 'text', 'datetime']

map_column_types = {
    'int': '(N)',
    'datetime': '(D)',
    'text': '(T)',
}

column_name_dict = {
    'int': [
        'price', 'amount', 'balance', 'total', 'cost', 'discount', 'tax', 'interest',
        'savings', 'inflation', 'revenue', 'investment', 'currency',
        'latitude', 'longitude', 'accuracy', 'score_percentage', 'rating_average',
        'growth_rate', 'exchange_rate', 'temperature', 'speed', 'altitude',
        'efficiency', 'power_usage', 'load_time', 'conversion_rate', 'profit_margin',
        'cpu_usage', 'memory_usage', 'click_through_rate', 'engagement_score', 'bounce_rate'
    ],
    'text': [
        'name', 'description', 'address', 'city', 'email', 'comment', 'product_name',
        'feedback', 'message', 'notes', 'status', 'details', 'subject', 'title',
        'first_name', 'last_name', 'username', 'phone_number', 'country', 'state',
        'language', 'bio', 'gender', 'occupation', 'company', 'url', 'hashtags',
        'category', 'filename', 'domain', 'browser', 'os', 'device_type', 'tag',
        'department', 'university', 'job_title', 'symptoms', 'diagnosis'
    ],
    'datetime': [
        'created_at', 'updated_at', 'birth_date', 'order_date', 'delivery_date',
        'transaction_date', 'due_date', 'last_modified', 'submitted_at', 'completion_date',
        'start_time', 'end_time', 'signup_date', 'login_time', 'logout_time',
        'publish_date', 'event_time', 'expiration_date', 'renewal_date',
        'interview_date', 'admission_date', 'discharge_date', 'deadline', 'timestamp',
        'last_seen', 'check_in', 'check_out', 'reported_at', 'reviewed_at'
    ]
}

flattened = [(column_type, column_name + f" {map_column_types[column_type]} ") for column_type in column_types for column_name in column_name_dict[column_type]]
# flattened = [(column_type, column_name) for column_type in column_types for column_name in column_name_dict[column_type]]


def generate_random_values(column_type):
    """Generate random data based on the column schema and any foreign key relationships."""
    if column_type == 'int':
        return random.randint(1, max_num_record)
        # return round(random.uniform(1.0, max_num_record), 2)
    elif column_type == 'text':
        return fake.text(max_nb_chars=20).replace('\n', '\\n')
    elif column_type == 'datetime':
        # Generate a fake datetime and format it as 'yyyy-MM-dd HH:mm:ss'
        return fake.date_time_this_century().strftime('%Y-%m-%d %H:%M:%S')


def generate_unique_values(datatype, n):
    """Efficiently generate n unique values based on the specified datatype."""
    if datatype == 'int':
        return random.sample(range(1, 10 * n), n)  # sample guarantees uniqueness

    if datatype == 'float':
        values = set()
        while len(values) < n:
            samples = np.round(np.random.uniform(1.0, 1000.0 * n, n * 10), 2)
            values.update(samples.tolist())
        return list(values)[:n]

    elif datatype == 'text':
        values = set()
        while len(values) < n:
            values.update(fake.text(max_nb_chars=20).replace('\n', '\\n') for _ in range(n))
        return list(values)[:n]

    elif datatype == 'datetime':
        start_date = datetime(1000, 1, 1)
        end_date = datetime(2025, 1, 1)
        delta = (end_date - start_date).days

        unique_days = random.sample(range(delta), n)
        return [start_date + timedelta(days=day) for day in unique_days]

def generate_random_schema(prev_primary_keys: dict):
    # choose columns randomly
    columns = random.sample(flattened, k = np.random.randint(min_columns, max_columns))
    # choose randomly primary keys
    num_pk  = np.random.randint(min_pk_columns - 1, max_pk_columns)
    pk_col_idx = np.random.choice(len(columns), size=(num_pk), replace=False)
    for i in pk_col_idx:
        columns[i] = (columns[i][0], columns[i][1] + " (P)")
    # add default primary key
    columns = [("int", "id (N) (P)")] + columns

    # choose forigen keys randomly
    if len(prev_primary_keys) != 0:
        num_fk      = np.random.randint(0, min(max_fk_columns, len(prev_primary_keys))) # num of forigen keys
        cols_fk     = np.random.choice(list(prev_primary_keys.keys()), size=(num_fk), replace=False)
    else:
        cols_fk = []
    # shuffle all columns to not have specific order
    return columns, list(cols_fk)
    

def create_csv_file(file_path, num_records, schema, table_name, foreign_keys, prev_primary_keys):
    """Create a CSV file for a table with random data based on the schema and foreign key relationships."""
    data = {}
    for column_type, column_name in tqdm(schema):
        # Remove any trailing spaces from column names
        clean_col_name = column_name.strip()
        if clean_col_name.endswith("(P)"):
            rows = generate_unique_values(column_type, num_records)
            prev_primary_keys[table_name + "_" + clean_col_name[:-4]] = rows
        else:
            rows = [generate_random_values(column_type) for _ in range(num_records)]
        data[clean_col_name] = rows
    
    for column_name in foreign_keys:
        clean_col_name = column_name.strip()
        data[clean_col_name] = random.choices(prev_primary_keys[column_name], k=num_records)
    
    df = pd.DataFrame(data)
    
    # First write to a string buffer to control formatting
    csv_buffer = df.to_csv(
        index=False,
        quoting=0,
        doublequote=False,
        escapechar='\\',
        lineterminator='\n',
        sep=','
    )
    
    # Remove spaces after commas and clean up the header line
    lines = csv_buffer.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove space after commas and trim trailing comma if exists
        cleaned_line = line.replace(', ', ',').rstrip(',')
        cleaned_lines.append(cleaned_line)
    
    # Write the cleaned content to file
    with open(file_path, 'w') as f:
        f.write('\n'.join(cleaned_lines))
    
    print(f"Finished generating {table_name}")


def create_random_tables(seed, folder_path, num_tables=5):
    """Create multiple random CSV tables with random schemas and relationships."""
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    prev_primary_keys = {} # colname: values # so we can choose from them 
    
    for i in range(num_tables):
        table_name = f"table_{i + 1}"
        schema, foreign_keys = generate_random_schema(prev_primary_keys)
        
        file_path = os.path.join(folder_path, f"{table_name}.csv")
        num_records = np.random.randint(min_num_records, max_num_record)
        create_csv_file(file_path, num_records, schema, table_name, foreign_keys, prev_primary_keys)

        # prev_primary_keys.extend(primary_keys)


create_random_tables(seed, folder_path, num_tables)
