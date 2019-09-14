import boto3
from botocore.exceptions import ClientError
import uuid
from io import BytesIO
import pickle
import pandas as pd
import syn_inflow.data_science_layer.utilities.exception_handling as excep


def create_bucket(bucket_str: str) -> str:
    s3 = boto3.resource('s3', region_name='us-west-2')
    bucket_name = ''.join([bucket_str, str(uuid.uuid4())])
    bucket_name = format_string_for_s3(bucket_name)
    a_bucket = s3.create_bucket(Bucket=bucket_name,
                                CreateBucketConfiguration={
                                    'LocationConstraint': 'us-west-2'},
                                )
    return a_bucket.name


def check_for_bucket_with_string(bucket_str: str) -> bool:
    s3 = boto3.resource('s3', region_name='us-west-2')
    available_buckets = s3.buckets.all()
    bucket_str = format_string_for_s3(bucket_str)
    for bucket in available_buckets:
        if bucket_str in bucket.name:
            return False
    return True


def get_bucket_id_with_string(bucket_str: str) -> str:
    s3 = boto3.resource('s3', region_name='us-west-2')
    available_buckets = s3.buckets.all()
    bucket_str = format_string_for_s3(bucket_str)
    for bucket in available_buckets:
        found, output = search_bucket(bucket, bucket_str, s3)
        if found:
            return output
    return ''


def check_if_object_exists(key, bucket_id):
    s3 = boto3.resource('s3')
    try:
        s3.Object(bucket_id, key).load()
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise
    return True


def remove_object_from_bucket(key, bucket_id):
    s3 = boto3.resource('s3')
    try:
        s3.Object(bucket_id, key).delete()
    except Exception as ex:
        print(ex)
        print('delete failed')


def get_bucket_with_id(bucket_id: str) -> object:
    s3 = boto3.resource('s3', region_name='us-west-2')
    available_buckets = s3.buckets.all()
    for bucket in available_buckets:
        if bucket_id == bucket.name:
            return bucket
    return None


def get_bucket_with_string(bucket_str: str) -> object:
    s3 = boto3.resource('s3', region_name='us-west-2')
    available_buckets = s3.buckets.all()
    bucket_str = format_string_for_s3(bucket_str)
    for bucket in available_buckets:
        if bucket_str in bucket.name:
            return bucket
    return None


def delete_bucket(a_bucket):
    for key in a_bucket.objects.all():
        key.delete()
    a_bucket.delete()


def format_string_for_s3(name: str) -> str:
    name = name.lower()
    name = name.replace('_', '-')
    return name


def load_pickle_from_s3(bucket_name, filename):
    objects = []
    s3 = boto3.resource('s3')
    with BytesIO() as data:
        s3.Bucket(bucket_name).download_fileobj(
            filename, data)
        data.seek(0)  # move back to the beginning after writing
        objects.append(pickle.load(data))
        if hasattr(objects[0], 'data'):
            data = objects[0].data
        else:
            data = objects[0]

    return data


@excep.function_raises_val(
    returns=pd.DataFrame(),
    exception_info='Load CSV from S3 failed',
    tag='AWS S3')
def read_csv_from_s3(bucket_name, filename):
    s3 = boto3.resource('s3')
    with BytesIO() as data:
        s3.Bucket(bucket_name).download_fileobj(
            filename, data)
        data.seek(0)  # move back to the beginning after writing
        data = pd.read_csv(data)

    return data


def write_pickle(data, name, bucket_id):
    s3_resource = boto3.resource('s3', region_name='us-west-2')
    pickle_buffer = pickle.dumps([data])
    s3_resource.Object(
        bucket_id, name + '.pkl').put(Body=pickle_buffer)


def read_csv_to_pandas(obj):
    return pd.read_csv(BytesIO(obj.get()['Body'].read()))


def search_bucket(bucket, bucket_str, s3):
    if bucket_str in bucket.name:
        try:
            s3.meta.client.head_bucket(Bucket=bucket.name)
        except Exception as ex:
            print(ex)
            # bucket.name = create_bucket(bucket_str)
        return True, bucket.name
    return False, ''
