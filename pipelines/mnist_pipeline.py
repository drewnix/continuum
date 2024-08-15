import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset
from components.download_mnist import download_mnist
from components.train_model import train_model
from components.serve_model import serve_model

# Define the pipeline
@dsl.pipeline(
    name="MNIST Pipeline",
    description="A pipeline that downloads MNIST data, trains a model, and serves it with KServe"
)
def mnist_pipeline():
    # Download MNIST data
    download_task = download_mnist()
    
    # Train the model
    train_task = train_model(data=download_task.outputs['output_data'])
    
    # Create KServe inference service
    # serve_task = create_kserve(model=train_task.outputs['output_model'], service_name="mnist-service")

# Compile the pipeline
kfp.compiler.Compiler().compile(mnist_pipeline, "mnist_pipeline2.yaml")