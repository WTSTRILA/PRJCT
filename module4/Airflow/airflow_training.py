from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime
import os
import zipfile
import boto3

IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
s3_data_path = 's3://bones-pipeline/Bone_Fracture_Binary_Classification.zip'

def download_and_extract_s3_zip(**kwargs):
    aws_access_key_id = BaseHook.get_connection('aws_default').login
    aws_secret_access_key = BaseHook.get_connection('aws_default').password

    if not aws_access_key_id or not aws_secret_access_key:
        raise EnvironmentError("AWS credentials are not set in connection.")

    local_zip_path = '/tmp/Bone_Fracture_Binary_Classification.zip'
    extraction_path = '/tmp/bone_data'
    os.makedirs(extraction_path, exist_ok=True)

    s3_bucket, s3_key = s3_data_path.replace("s3://", "").split("/", 1)

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    try:
        s3_client.download_file(s3_bucket, s3_key, local_zip_path)
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        raise

    try:
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
    except zipfile.BadZipFile as e:
        print(f"Error unpacking zip file: {e}")
        raise

    kwargs['ti'].xcom_push(key='train_data', value=os.path.join(extraction_path, 'train'))
    kwargs['ti'].xcom_push(key='test_data', value=os.path.join(extraction_path, 'test'))
    kwargs['ti'].xcom_push(key='val_data', value=os.path.join(extraction_path, 'val'))


def create_train_and_evaluate_model(**kwargs):
    import tensorflow as tf

    train_data_path = kwargs['ti'].xcom_pull(key='train_data')
    val_data_path = kwargs['ti'].xcom_pull(key='val_data')
    test_data_path = kwargs['ti'].xcom_pull(key='test_data')
    learning_rate = kwargs.get('learning_rate', 0.001)

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_set = val_test_datagen.flow_from_directory(
        val_data_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_set = val_test_datagen.flow_from_directory(
        test_data_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_set, validation_data=val_set, epochs=10)

    test_loss, test_accuracy = model.evaluate(test_set)

    model.save('/tmp/bone_fracture_model', save_format='tf')
    print(f"Test accuracy: {test_accuracy}")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG(
        dag_id='bone_fracture_classification_dag',
        default_args=default_args,
        schedule_interval='@once',
        catchup=False,
) as dag:
    download_task = PythonOperator(
        task_id='download_and_extract_s3_zip',
        python_callable=download_and_extract_s3_zip,
        provide_context=True
    )

    train_task = PythonOperator(
        task_id='create_train_and_evaluate_model',
        python_callable=create_train_and_evaluate_model,
        provide_context=True,
        op_kwargs={'learning_rate': 0.001}
    )

    download_task >> train_task
