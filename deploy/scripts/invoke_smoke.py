import boto3, json, os
smr = boto3.client("sagemaker-runtime", region_name=os.environ.get("AWS_REGION","ap-southeast-1"))
ep = os.environ.get("ENDPOINT_NAME","aqi-xgb-endpoint")

payload = {
  "station_id":"VN_Hanoi_001","ts":1731324000.0,
  "pm25":52.0,"pm25_lag_1h":49.5,"pm25_roll_mean_6h":47.8,
  "temp":29.5,"humidity":72,"wind_speed":3.1,"wind_deg":210,
  "hour":14,"dow":2,"is_weekend":0,"month":11
}
resp = smr.invoke_endpoint(EndpointName=ep, Body=json.dumps(payload), ContentType="application/json")
print(resp["Body"].read().decode("utf-8"))
