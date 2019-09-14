
import boto3
import pickle
from io import StringIO
from .output_target import OutputTarget
from crcdal.output_layer.utilities.aws_s3_funcs import \
    check_for_bucket_with_string, get_bucket_id_with_string, create_bucket
from syn_inflow.data_science_layer.utilities.cache_path import get_cache_path


class AwsS3OutputTarget(OutputTarget):
    """Output target for a table pipeline to send data to an AWS S3 bucket"""

    def __init__(self):
        self.bucket_id = None
        self.s3_resource = None

    @classmethod
    def write_to_target(cls, data, name):
        self = cls()
        self.get_bucket_id()
        self.select_output(data, name)

    def get_bucket_id(self):
        # Get Configuration folder
        bucket_str = get_cache_path()
        bucket_str = bucket_str + "output"

        # Check if S3 Bucket exists, create it if not
        if check_for_bucket_with_string(bucket_str):
            self.bucket_id = create_bucket(bucket_str)
        else:
            self.bucket_id = get_bucket_id_with_string(bucket_str)

        return self.bucket_id

    def select_output(self, data, name):
        self.s3_resource = boto3.resource('s3')
        if self.format == 'csv':
            self.write_csv(data, name)
        elif self.format == 'pkl':
            self.write_pickle(data, name)
        elif self.format == 'json':
            self.write_json(data, name)

    def write_pickle(self, data, name):
        pickle_buffer = pickle.dumps([data])
        self.s3_resource.Object(
           self.bucket_id, name + '.pkl').put(Body=pickle_buffer)

    def write_csv(self, data, name):
        csv_buffer = StringIO()
        data.to_csv(csv_buffer)
        self.s3_resource.Object(self.bucket_id, name + '.csv').put(Body=csv_buffer.getvalue())

    def write_json(self, data, name):
        stringBuf = StringIO()
        data.to_json(stringBuf)
        self.s3_resource.Object(
            self.bucket_id, name + '.json').put(Body=stringBuf.getvalue())
