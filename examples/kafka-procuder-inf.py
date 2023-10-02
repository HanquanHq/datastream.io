from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import time
import random

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda m: json.dumps(m).encode('ascii'))

current_time = int(time.time())
i = 0
anomaly_cnt = 0
while True:
    try:
        if anomaly_cnt > 0:
            anomaly_cnt = anomaly_cnt - 1;
            random_speed = random.randint(100, 1000)
        else:
            if i % random.randint(30, 40) == 0:
                random_speed = random.randint(0, 1000)
                anomaly_cnt = random.randint(0, 6)
            else:
                random_speed = random.randint(0, 300)

        message = {"time": current_time + i, "speed": random_speed}
        print('sending {}'.format(message))
        producer.send('test1', message)
        time.sleep(0.1)
        i = i + 1
    except KafkaError:
        print('error')
        pass
