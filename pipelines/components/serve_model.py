from kfp.v2.dsl import component, Input, Dataset


# Define component to create KServe inference service
@component(base_image='python:3.9')
def serve_model(model: Input[Dataset], service_name: str):
    from kubernetes import client, config
    from kserve import KServeClient
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1TFServingSpec
    
    # Load Kubernetes configuration
    config.load_incluster_config()
    
    # Create KServe client
    kserve_client = KServeClient()
    
    # Define the inference service
    isvc = V1beta1InferenceService(
        api_version="serving.kubeflow.org/v1beta1",
        kind="InferenceService",
        metadata=client.V1ObjectMeta(name=service_name, namespace="default"),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                tensorflow=V1beta1TFServingSpec(
                    storage_uri=model.path
                )
            )
        )
    )
    
    # Create the inference service
    kserve_client.create(isvc)
