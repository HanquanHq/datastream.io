from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import time
import random

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda m: json.dumps(m).encode('ascii'))

current_time = int(time.time())
i = 0
current_speed = random.randint(0, 100)  # 初始速度值
time_in_range = random.randint(10, 30)  # 初始速度在一个范围内保持的时间

while True:
    try:
        if i % time_in_range == 0:
            time_in_range = random.randint(10, 30)  # 在一个范围内保持时间
            target_speed = random.randint(0, 100)  # 生成新的目标速度
        # 逐渐接近目标速度
        if current_speed < target_speed:
            current_speed += 1
        elif current_speed > target_speed:
            current_speed -= 1
        message = {"time": current_time + i, "speed": current_speed}
        print('sending {}'.format(message))
        producer.send('test1', message)
        time.sleep(0.1)
        i = i + 1
    except KafkaError:
        print('error')
        pass
