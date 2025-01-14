"""
An abstract base class for live models that can run together with VL-InterpreT.

To run your own model with VL-InterpreT, create another file your_model.py in this
folder that contains a class Your_Model (use title case for the class name), which
inherits from the VL_Model class and implements the data_setup method. The data_setup
method should take the ID, image and text of a given example, run a forward pass for
this example with your model, and return the corresponding attention, hidden states
and other required data that can be visualized with VL-InterpreT.
"""

from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import urllib.request


class VL_Model(ABC):
    """
    To run a live transformer with VL-InterpreT, define your own model class by inheriting
    from this class and implementing the data_setup method.

    Please follow these naming patterns to make sure your model runs easily with VL-InterpreT:
        - Create a new python script in this folder for your class, and name it in all lower
          case (e.g., yourmodelname.py)
        - Name your model class in title case, e.g., Yourmodelname. This class name should be
          the result of calling 'yourmodelname'.title(), where 'yourmodelname.py' is the name
          of your python script.

    Then you can run VL-InterpreT with your model:
        python run_app.py -p 6006 -d example_database2 -m yourmodelname your_model_parameters
    """

    @abstractmethod
    def data_setup(self, example_id: int, image_location: str, input_text: str) -> dict:
        """
        This method should run a forward pass with your model given the input image and
        text, and return the required data. See app/database/db_example.py for specifications
        of the return data format, and see the implementation in kdvlp.py for an example.
        """
        return {
            "ex_id": example_id,
            "image": np.array(),
            "tokens": [],
            "txt_len": 0,
            "img_coords": [],
            "attention": np.array(),
            "hidden_states": np.array(),
        }

    def fetch_image(self, image_location: str):
        """
        This helper function takes the path to an image (either an URL or a local path) and
        returns the image as an numpy array.
        """
        if image_location.startswith("http"):
            urllib.request.urlretrieve(image_location, "temp.jpg")
            image_location = "temp.jpg"

        img = Image.open(image_location).convert("RGB")
        img = np.array(img)
        return img


from transformers import MBartForCausalLM
from transformers import DonutProcessor, VisionEncoderDecoderModel
import dagshub
import mlflow
import time
import os
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()


class Donut(VL_Model):

    def __init__(self, device="cuda"):
        self.device = torch.device(device, 0)
        self.device = "cpu"
        print(f"Device: {self.device}")
        # self.processor, self.model = self.fetch_model_from_mlflow()
        self.processor, self.model = (
            self.fetch_model_from_local()
        )  # self.fetch_model_from_mlflow()
        # if device == "cuda":
        # torch.cuda.set_device(0)
        # self.model, self.tokenizer = self.build_model(ckpt_file)

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
        # from kaggle_secrets import UserSecretsClient
        # user_secrets = UserSecretsClient()
        # token = user_secrets.get_secret("dags_hub_token")
        #         from google.colab import userdata
        #         token = userdata.get('dags_hub_token')
        token = os.getenv("dags_hub_token")

        dagshub.auth.add_app_token(token)

        dagshub.init(
            repo_owner="zaheramasha",
            repo_name="Finetuning_paligemma_Zaka_capstone",
            mlflow=True,
        )

        # Define the MLflow run ID and artifact path
        run_id = "c41cfd149a8c44f3a92d8e0f1253af35"  # Donut model trained on the PyvizAndMarkMap dataset for 27 epochs reaching a train loss of 0.168
        run_id = "89bafd5e525a4d3e9d004e13c9574198"  # Donut model trained on the PyvizAndMarkMap dataset for 27 + 51 = 78 epochs reaching a train loss of 0.0353. This run was a continuation of the 27 epoch one
        # in reality the model saved here is trained for 20 + 50 = 70 epochs, since we are saving the model every 10 epochs.
        run_id = "35c97d004ddd4b5ca4cdb7737c6d6369"  # for the model in the 4th in the sequence
        artifact_path = "Donut_model/model"

        # Create the model URI using the run ID and artifact path
        model_uri = f"runs:/{run_id}/{artifact_path}"
        print(
            mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path=artifact_path)
        )
        # Load the model and processors from the MLflow artifact
        # loaded_model_bundle = mlflow.transformers.load_model(artifact_path=artifact_path, run_id=run_id)
        # for the 20 epochs trained model
        model_uri = f"mlflow-artifacts:/0a5d0550f55c4169b80cd6439556be8b/c41cfd149a8c44f3a92d8e0f1253af35/artifacts/Donut_model"

        # for the fully 70 epochs trained model
        model_uri = f"mlflow-artifacts:/17c375f6eab34c63b2a2e7792803132e/89bafd5e525a4d3e9d004e13c9574198/artifacts/Donut_model"
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

    def data_setup(self, ex_id: int, image_location: str, input_text: str) -> dict:
        # pixel_values, labels, answers = batch
        pixel_values = self.processor(
            image_location, random_padding=False, return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.squeeze()

        pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension

        batch_size = pixel_values.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            device=self.device,
        )

        # put the inputs on the device
        pixel_values = pixel_values.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        print(f"pixel_values: {pixel_values.shape}")
        print(f"decoder_input_ids: {decoder_input_ids.shape}")

        # decoder_input_ids = decoder_input_ids.view(1, -1)  # Reshape to [1, 3]
        # print('decoder_input_ids: ', decoder_input_ids.shape)  # Output should be: torch.Size([1, 3])
        with torch.no_grad():
            outputs = self.model(
                pixel_values,
                decoder_input_ids,
                output_hidden_states=True,
                output_attentions=True,
            )  # labels=labels)

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=2500,  # self.config.get("max_length", 512),
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        predictions = self.processor.tokenizer.batch_decode(outputs.sequences)

        pixel_values = pixel_values.cpu()
        decoder_input_ids = decoder_input_ids.cpu()

        return outputs, predictions


donut_test = Donut()
