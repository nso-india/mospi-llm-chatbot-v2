s3 = boto3.client("s3")

def list_s3_files(bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    files = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" not in page:
            continue
        for item in page["Contents"]:
            key = item["Key"]
            if key.lower().endswith((".pdf", ".xlsx", ".xls")):
                files.append(key)
    
    print(files)