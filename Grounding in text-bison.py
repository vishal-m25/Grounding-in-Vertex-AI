import vertexai
from vertexai.language_models import TextGenerationModel, GroundingSource
from google.oauth2.service_account import Credentials

key_path='project-411006-610b3f7622fe.json' ## Replace with your json file name
credentials = Credentials.from_service_account_file(
    key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform'])

PROJECT_ID = 'Text-bison-grounding'  ## Replace with your project ID
REGION = 'us-central1'
data_store_id="Data-store" ## Replace with datastore id that you create in vertex ai search and conversation
location="global"

vertexai.init(project = PROJECT_ID, location = REGION, credentials = credentials)
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.9,
    "grounding_source":GroundingSource.VertexAISearch(data_store_id=data_store_id,location=location),
    "top_p": 1,
}
model = TextGenerationModel.from_pretrained("text-bison")  ## The model can be changed but make sure it has the grounding option 
response = model.predict(
    """ what are clouds made off? """,
    **parameters
)
print(f"Response from Model: {response.text}")