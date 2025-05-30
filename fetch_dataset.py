!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="SkpACeECQkUHe2S3K5rw")
project = rf.workspace("mamytest").project("test_smoke_fire_model")
version = project.version(1)
dataset = version.download("yolov8")




## raw url - https://app.roboflow.com/ds/rX8FkZtKWa?key=vlvAj87KmG

# cmd  curl -L "https://app.roboflow.com/ds/rX8FkZtKWa?key=vlvAj87KmG" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
                