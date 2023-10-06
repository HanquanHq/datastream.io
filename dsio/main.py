"""
1. Reads an input data stream that consists of sensor data
2. Generates Kibana and/or Bokeh dashboards to visualize the data stream
3. Applies scores to the incoming sensor data, using selected anomaly detector
4. Restreams input data and scores to ElasticSearch and/or Bokeh server
"""

import sys
import datetime
import time
import threading
import math
import webbrowser

from queue import Queue

import numpy as np
import pandas as pd

from kafka import KafkaConsumer
import json

from bokeh.plotting import curdoc

from .restream.elastic import init_elasticsearch
from .restream.elastic import upload_dataframe
from .dashboard.kibana import generate_dashboard as generate_kibana_dashboard
from .dashboard.bokeh import generate_dashboard as generate_bokeh_dashboard

from .helpers import parse_arguments, normalize_timefield
from .helpers import select_sensors, init_detector_models, load_detector

from .exceptions import DsioError

MAX_BATCH_SIZE = 200
doc = curdoc()


def restream_dataframe(
        dataframe, detector, sensors=None, timefield=None,
        speed=10, es_uri=None, kibana_uri=None, index_name='',
        entry_type='', bokeh_port=5001, cols=3):
    """
        Restream selected sensors & anomaly detector scores from an input
        pandas dataframe to an existing Elasticsearch instance and/or to a
        built-in Bokeh server.

        Generates respective Kibana & Bokeh dashboard apps to visualize the
        stream in the browser
    """

    dataframe, timefield, available_sensors = normalize_timefield(
        dataframe, timefield, speed
    )

    dataframe, sensors = select_sensors(
        dataframe, sensors, available_sensors, timefield
    )

    if es_uri:
        es_conn = init_elasticsearch(es_uri)
        # Generate dashboard with selected fields and scores
        generate_kibana_dashboard(es_conn, sensors, index_name)
        webbrowser.open(kibana_uri + '#/dashboard/%s-dashboard' % index_name)
    else:
        es_conn = None

    # Queue to communicate between restreamer and dashboard threads
    update_queue = Queue()
    if bokeh_port:
        generate_bokeh_dashboard(
            sensors, title=detector.__name__, cols=cols,
            port=bokeh_port, update_queue=update_queue
        )

    restream_thread = threading.Thread(
        target=threaded_restream_dataframe,
        args=(dataframe, sensors, detector, timefield, es_conn,
              index_name, entry_type, bokeh_port, update_queue)
    )
    restream_thread.start()


def threaded_restream_dataframe(dataframe, sensors, detector, timefield,
                                es_conn, index_name, entry_type, bokeh_port,
                                update_queue, interval=3, sleep_interval=1):
    """ Restream dataframe to bokeh and/or Elasticsearch """
    # Split data into batches
    batches = np.array_split(dataframe, math.ceil(dataframe.shape[0] / MAX_BATCH_SIZE))
    print("batchs size is: " + str(len(batches)))

    # Initialize anomaly detector models, train using first batch
    models = init_detector_models(sensors, batches[0], detector)
    print("models is: " + str(
        models))  # models is: {'accelerator_pedal_position': Gaussian1D(), 'engine_speed': Gaussian1D()}

    first_pass = True
    for batch in batches:
        for sensor in sensors:  # Apply the scores
            batch['SCORE_{}'.format(sensor)] = models[sensor].score_anomaly(batch[sensor])
            batch['FLAG_{}'.format(sensor)] = models[sensor].flag_anomaly(batch[sensor])

        end_time = np.min(batch[timefield])
        recreate_index = first_pass

        while end_time < np.max(batch[timefield]):  # 将这一秒的数据送进update_queue，用来检测异常、展示图像
            start_time = end_time
            end_time += interval * 1000
            if not recreate_index:
                while np.round(time.time()) < end_time / 1000.:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    time.sleep(sleep_interval)

            ind = np.logical_and(batch[timefield] <= end_time,
                                 batch[timefield] > start_time)
            print('\nWriting {} rows dated {} to {}'
                  .format(np.sum(ind),
                          datetime.datetime.fromtimestamp(start_time / 1000.),
                          datetime.datetime.fromtimestamp(end_time / 1000.)))

            if bokeh_port:  # 将这一秒的数据放入更新队列，用于 Bokeh 更新
                update_queue.put(batch.loc[ind])

            if es_conn:  # Stream batch to Elasticsearch
                upload_dataframe(es_conn, batch.loc[ind], index_name, entry_type,
                                 recreate=recreate_index)
            recreate_index = False

        if first_pass:
            for sensor in sensors:  # Fit the models
                models[sensor].fit(batch[sensor])
        else:
            for sensor in sensors:  # Update the models
                models[sensor].update(batch[sensor])
        first_pass = False


def restream_kafka(topic, bootstrap_servers, auto_offset_reset,
                   detector, sensors=None, timefield=None,
                   speed=10, index_name='',
                   entry_type='', bokeh_port=5001, cols=3):
    # Queue to communicate between restreamer and dashboard threads
    update_queue = Queue()

    if bokeh_port:
        generate_bokeh_dashboard(
            sensors, title=detector.__name__, cols=cols,
            port=bokeh_port, update_queue=update_queue
        )

    restream_thread = threading.Thread(
        target=threaded_restream_kafka,
        args=(topic, bootstrap_servers, sensors, detector, timefield,
              index_name, entry_type, bokeh_port,
              update_queue, auto_offset_reset, cols)
    )
    restream_thread.start()


def threaded_restream_kafka(topic, bootstrap_servers, sensors, detector, timefield,
                            index_name, entry_type, bokeh_port,
                            update_queue, auto_offset_reset, cols, interval=3, sleep_interval=1, ):
    print("start threaded_restream_kafka")
    print("auto_offset_reset is: " + auto_offset_reset)
    TRAIN_SIZE = 200
    consumer = KafkaConsumer(topic,
                             group_id='dsio-group',
                             bootstrap_servers=bootstrap_servers,
                             # decoding json messages
                             value_deserializer=lambda m: json.loads(m.decode('ascii')),
                             auto_offset_reset=auto_offset_reset)

    # 收集训练集
    try:
        to_train = []
        while True:
            received = consumer.poll(timeout_ms=1000)
            print("received training data: " + str(received) + ", size " + str(len(received)))
            for topic_partition, messages in received.items():
                for m in messages:
                    print(m.value)
                    to_train.append(m.value)
            if len(to_train) < TRAIN_SIZE:
                pass
            else:
                print("to_train is below")
                print(to_train)
                batch = pd.DataFrame(to_train)
                models = init_detector_models(sensors, batch, detector)
                print("models is: " + str(
                    models))  # models is: {'accelerator_pedal_position': Gaussian1D(), 'engine_speed': Gaussian1D()}
                for sensor in sensors:  # Fit the models
                    models[sensor].fit(batch[sensor])
                break
    except KeyboardInterrupt:
        print('Stopping Kafka consumer...')
        consumer.close()

    print("training data finish")
    # 收集测试集
    try:
        to_analyze = []
        while True:
            received = consumer.poll(timeout_ms=1000)
            print("received test data: " + str(received) + ", size " + str(len(received)))
            for topic_partition, messages in received.items():
                for m in messages:
                    # print(m.value)
                    to_analyze.append(m.value)
            if len(to_analyze) < 20:
                pass
            else:
                print("to_analyze is below")
                print(to_analyze)
                batch = pd.DataFrame(to_analyze)

                for sensor in sensors:  # Apply the scores
                    batch['SCORE_{}'.format(sensor)] = models[sensor].score_anomaly(batch[sensor])
                    batch['FLAG_{}'.format(sensor)] = models[sensor].flag_anomaly(batch[sensor])
                    # update_queue.put(batch.loc[ind])
                    update_queue.put(batch)

                to_analyze = []
    except KeyboardInterrupt:
        print('Stopping Kafka consumer...')
    finally:
        consumer.close()


def main():
    """ Main function """
    args = parse_arguments()

    try:
        detector = load_detector(args.detector, args.modules)

        # Generate index name from input filename
        index_name = args.input.split('/')[-1].split('.')[0].split('_')[0]
        if not index_name:
            index_name = 'dsio'

        if args.kafka_uri:
            print('Kafka input mode...')
            restream_kafka(
                topic=args.input,
                bootstrap_servers=args.kafka_uri.split(';'),
                auto_offset_reset=args.auto_offset_reset,
                detector=detector,
                sensors=args.sensors, timefield=args.timefield,
                index_name=index_name,
                entry_type=args.entry_type, bokeh_port=int(args.bokeh_port),
            )
        else:
            print('Loading the data from csv...')
            dataframe = pd.read_csv(args.input, sep=',')
            print('Done.\n')

            restream_dataframe(
                dataframe=dataframe, detector=detector,
                sensors=args.sensors, timefield=args.timefield,
                speed=int(float(args.speed)), es_uri=args.es and args.es_uri,
                kibana_uri=args.kibana_uri, index_name=index_name,
                entry_type=args.entry_type, bokeh_port=int(args.bokeh_port),
                cols=int(args.cols)
            )

    except DsioError as exc:
        print(repr(exc))
        sys.exit(exc.code)


if __name__ == '__main__':
    main()
