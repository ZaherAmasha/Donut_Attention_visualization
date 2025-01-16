# Use this MBartForCausalLM and VisionEncoderDecoderModel to access the transformers source code that defines the Donut model
from transformers import MBartForCausalLM
from transformers import DonutProcessor, VisionEncoderDecoderModel
import dagshub
import mlflow
import os
import torch
from dotenv import load_dotenv

load_dotenv()


class Donut:

    def __init__(self, device="cuda"):
        self.device = torch.device(device, 0)
        self.device = "cpu"
        print(f"Device: {self.device}")
        # self.processor, self.model = self.fetch_model_from_mlflow()
        self.processor, self.model = self.fetch_model_from_local()

    def fetch_model_from_local(self):
        local_model_path = "/home/zaher/Projects/visualizing_Donut_attention/loaded_model/Donut_model_49"

        # Load the model and processor directly from the local path
        loaded_model_bundle = mlflow.transformers.load_model(
            model_uri=local_model_path, device=self.device
        )

        model = loaded_model_bundle.model
        processor = DonutProcessor(
            tokenizer=loaded_model_bundle.tokenizer,
            feature_extractor=loaded_model_bundle.feature_extractor,
            image_processor=loaded_model_bundle.image_processor,
        )

        print(model.config.encoder.image_size)
        print(model.config.decoder.max_length)
        return processor, model

    def fetch_model_from_mlflow(self):

        token = os.getenv("dags_hub_token")

        dagshub.auth.add_app_token(token)

        dagshub.init(
            repo_owner="zaheramasha",
            repo_name="Finetuning_paligemma_Zaka_capstone",
            mlflow=True,
        )

        # Define the MLflow run ID and artifact path
        # run_id = "c41cfd149a8c44f3a92d8e0f1253af35"  # Donut model trained on the PyvizAndMarkMap dataset for 27 epochs reaching a train loss of 0.168
        # run_id = "89bafd5e525a4d3e9d004e13c9574198"  # Donut model trained on the PyvizAndMarkMap dataset for 27 + 51 = 78 epochs reaching a train loss of 0.0353. This run was a continuation of the 27 epoch one
        # in reality the model saved here is trained for 20 + 50 = 70 epochs, since we are saving the model every 10 epochs.
        run_id = "35c97d004ddd4b5ca4cdb7737c6d6369"  # for the model in the 4th in the sequence

        artifact_path = "Donut_model/model"

        # Create the model URI using the run ID and artifact path
        model_uri = f"runs:/{run_id}/{artifact_path}"
        print(
            mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path=artifact_path)
        )
        # Load the model and processors from the MLflow artifact
        # for the 20 epochs trained model
        # model_uri = f"mlflow-artifacts:/0a5d0550f55c4169b80cd6439556be8b/c41cfd149a8c44f3a92d8e0f1253af35/artifacts/Donut_model"
        # for the fully 70 epochs trained model
        # model_uri = f"mlflow-artifacts:/17c375f6eab34c63b2a2e7792803132e/89bafd5e525a4d3e9d004e13c9574198/artifacts/Donut_model"
        # for the 4th in the sequence
        model_uri = f"mlflow-artifacts:/17c375f6eab34c63b2a2e7792803132e/35c97d004ddd4b5ca4cdb7737c6d6369/artifacts/Donut_model_49"

        loaded_model_bundle = mlflow.transformers.load_model(
            model_uri=model_uri,
            device=self.device,
            dst_path="/home/zaher/Projects/visualizing_Donut_attention/loaded_model",  # "cpu"
        )  #'cuda')

        model = loaded_model_bundle.model
        processor = DonutProcessor(
            tokenizer=loaded_model_bundle.tokenizer,
            feature_extractor=loaded_model_bundle.feature_extractor,
            image_processor=loaded_model_bundle.image_processor,
        )
        print(model.config.encoder.image_size)
        print(model.config.decoder.max_length)
        return processor, model


donut_test = Donut()
