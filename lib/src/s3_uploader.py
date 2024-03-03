import os

import boto3


class UploaderS3(object):
    def __init__(self):
        self.bucket_name = os.environ['BUCKET_NAME']
        self.session = boto3.session.Session()
        self.s3_client = self.session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net'
        )

    def upload_to_s3(self, file_name_from: str, file_name_to: str):
        self.s3_client.upload_file(
            f'{file_name_from}',
            self.bucket_name,
            f'empty/{file_name_to}'
        )
        return None
