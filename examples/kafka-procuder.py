from kafka import KafkaProducer
from kafka.errors import KafkaError
import json

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda m: json.dumps(m).encode('ascii'))

messages = [
    {"time":1522513435, "speed": 30},
    {"time":1522513535, "speed": 32},
    {"time":1522513635, "speed": 31},
    {"time":1522513735, "speed": 29},
    {"time":1522513835, "speed": 32},
    {"time":1522513935, "speed": 30},
    {"time":1522514415, "speed": 600},
    {"time":1522514425, "speed": 30},
    {"time":1522514435, "speed": 31},
    {"time":1522514455, "speed": 29},
]

for m in messages:
    try:
        print('sending {}'.format(m))
        future=producer.send('test1', m)
        record_metadata = future.get(timeout=10)
    except KafkaError:
        # Decide what to do if produce request failed...
        log.exception()
        print('error')
        pass
