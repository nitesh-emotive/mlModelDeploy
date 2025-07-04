# import os
# import torch
# import zipfile
# import requests
# import tempfile
# import shutil
# import io
# import pandas as pd
# from flask import Flask, request, jsonify, send_file
# from werkzeug.utils import secure_filename
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from tqdm import tqdm
# import re
# import logging
# from datetime import datetime

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)

# # ---------- CONFIG ----------
# MODEL_PATH = "fine_tuned_model_MPN8"  # Update this path
# EMAIL = "user@example.com"       # Replace with your email
# PASSWORD = "securePassword123"   # Replace with your password
# LOGIN_URL = 'https://towbarinstructions.co.uk/api/login'
# UPLOAD_URL = 'https://towbarinstructions.co.uk/api/uploadFileDownloadMvl/article/upload'

# # Flask config
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# # Global model variables
# tokenizer = None
# model = None
# device = None

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def load_model():
#     """Load the ML model once at startup"""
#     global tokenizer, model, device
#     try:
#         logger.info("Loading T5 model...")
#         tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
#         model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
#         # model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, from_safetensors=True)

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#         model.eval()
#         logger.info(f"Model loaded successfully on {device}")
#         return True
#     except Exception as e:
#         logger.error(f"Failed to load model: {e}")
#         return False

# def detect_title_column(df):
#     """Detect the title column in the dataframe"""
#     for col in df.columns:
#         if col.lower().strip() in ['title', 'product title', 'product_title']:
#             return col
#     raise ValueError("No title or product_title column found!")

# def predict_batch(titles):
#     """Predict MPNs for a batch of titles"""
#     batch_size = 64 if torch.cuda.is_available() else 4
#     predictions = []
    
#     for i in tqdm(range(0, len(titles), batch_size), desc="Extracting MPNs"):
#         batch = titles[i:i + batch_size]
#         prompts = [f"extract only MPN and OEM numbers not a model no. from this automotive part title: {t}" for t in batch]
#         inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
#         with torch.no_grad():
#             outputs = model.generate(inputs['input_ids'], max_new_tokens=20, num_beams=4, early_stopping=True)
        
#         decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         predictions.extend(decoded)
    
#     return predictions

# def clean_mpns(mpn_string):
#     """Clean and format MPN strings"""
#     if pd.isna(mpn_string) or not mpn_string:
#         return ""
#     mpn_string = re.sub(r'[\/;]', ',', mpn_string)
#     mpns = mpn_string.split(',')
#     return '|'.join([m.strip() for m in mpns if len(m.strip()) > 4 and not m.strip().isalpha() and '$' not in m])

# def explode_mpns(df):
#     """Explode MPN column to separate rows"""
#     if "MPN" not in df.columns:
#         return df
#     return df.assign(MPN=df["MPN"].str.split("|")).explode("MPN").reset_index(drop=True)

# def extract_mpn_from_file(file_path):
#     """Extract MPNs from uploaded file"""
#     try:
#         # Read file based on extension
#         ext = os.path.splitext(file_path)[1].lower()
#         if ext == '.csv':
#             df = pd.read_csv(file_path)
#         else:
#             df = pd.read_excel(file_path)
        
#         # Detect title column
#         title_col = detect_title_column(df)
        
#         # Predict MPNs
#         logger.info("Starting MPN prediction...")
#         df['predicted_mpn'] = predict_batch(df[title_col].astype(str).tolist())
#         df['MPN'] = df['predicted_mpn'].apply(clean_mpns)
        
#         # Explode MPNs
#         df = explode_mpns(df)
        
#         return df
#     except Exception as e:
#         logger.error(f"Error extracting MPNs: {e}")
#         raise

# def upload_to_fitment_api(csv_content):
#     """Upload CSV content to fitment API and get ZIP response"""
#     try:
#         # Login
#         logger.info("Logging in to Fitment API...")
#         response = requests.post(LOGIN_URL, json={"email": EMAIL, "password": PASSWORD})
#         if response.status_code != 200:
#             raise Exception(f"Login failed: {response.text}")
        
#         token = response.json().get("token")
        
#         # Upload file
#         logger.info("Uploading file to Fitment API...")
#         files = {'file': ('data.csv', csv_content, 'text/csv')}
#         headers = {"accept": "application/zip", "Authorization": token}
#         r = requests.post(UPLOAD_URL, headers=headers, files=files)
        
#         if r.status_code == 200 and 'zip' in r.headers.get('Content-Type', ''):
#             logger.info("ZIP received from API")
#             return r.content
#         else:
#             raise Exception(f"Failed to get fitment ZIP: {r.status_code} {r.text}")
            
#     except Exception as e:
#         logger.error(f"Error uploading to fitment API: {e}")
#         raise

# def transform_fitment_data_to_expected_format(zip_content):
#     """Transform the fitment data from ZIP to expected CSV format"""
#     try:
#         logger.info("Transforming fitment data to expected format...")
        
#         # Extract CSV from ZIP
#         with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_file:
#             csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
#             if not csv_files:
#                 raise Exception("No CSV file found in ZIP")
            
#             # Read the first CSV file
#             with zip_file.open(csv_files[0]) as csv_file:
#                 df = pd.read_csv(csv_file)
        
#         logger.info(f"Original data shape: {df.shape}")
#         logger.info(f"Original columns: {df.columns.tolist()}")
        
#         # Group by MPN to handle multiple compatibilities per item
#         transformed_rows = []
        
#         # Group data by MPN
#         grouped = df.groupby('mpn')
        
#         for mpn, group in grouped:
#             # Skip if MPN is invalid
#             if pd.isna(mpn) or str(mpn) == 'nan':
#                 continue
            
#             # Add the first row with Revise action and ItemID
#             first_row = {
#                 'Action': 'Revise',
#                 'ItemID': str(mpn),
#                 'Relationship': '',
#                 'RelationshipDetails': ''
#             }
#             transformed_rows.append(first_row)
            
#             # Add compatibility rows for each record in the group
#             for _, row in group.iterrows():
#                 make = str(row.get('make', ''))
#                 model = str(row.get('model', ''))
#                 variant = str(row.get('variant', ''))
#                 engine = str(row.get('engine', ''))
#                 body_style = str(row.get('body_style', ''))
#                 type_field = str(row.get('type', ''))
#                 year = str(row.get('year', ''))
                
#                 # Create relationship details string
#                 relationship_details = []
#                 if make != 'nan' and make: relationship_details.append(f"Make={make}")
#                 if model != 'nan' and model: relationship_details.append(f"Model={model}")
#                 if variant != 'nan' and variant: relationship_details.append(f"Variant={variant}")
#                 if body_style != 'nan' and body_style: relationship_details.append(f"BodyStyle={body_style}")
#                 if type_field != 'nan' and type_field: relationship_details.append(f"Type={type_field}")
#                 if engine != 'nan' and engine: relationship_details.append(f"Engine={engine}")
#                 if year != 'nan' and year: relationship_details.append(f"Cars Year={year}")
                
#                 relationship_details_str = "|".join(relationship_details)
                
#                 # Create compatibility row
#                 compatibility_row = {
#                     'Action': 'Add',
#                     'ItemID': '',
#                     'Relationship': 'Compatibility',
#                     'RelationshipDetails': f"Car {relationship_details_str}"
#                 }
                
#                 transformed_rows.append(compatibility_row)
        
#         # Create DataFrame from transformed rows
#         transformed_df = pd.DataFrame(transformed_rows)
        
#         logger.info(f"Transformed data shape: {transformed_df.shape}")
#         logger.info(f"Sample transformed data:\n{transformed_df.head()}")
        
#         return transformed_df
        
#     except Exception as e:
#         logger.error(f"Error transforming fitment data: {e}")
#         raise

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy',
#         'model_loaded': model is not None,
#         'timestamp': datetime.now().isoformat()
#     })

# @app.route('/process-mpn', methods=['POST'])
# def process_mpn():
#     """Main endpoint to process CSV and return transformed CSV"""
#     try:
#         # Check if file is present
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file provided'}), 400
        
#         file = request.files['file']
        
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
        
#         if not allowed_file(file.filename):
#             return jsonify({'error': 'Invalid file type. Only CSV, XLS, XLSX allowed'}), 400
        
#         # Check if model is loaded
#         if model is None:
#             return jsonify({'error': 'Model not loaded'}), 500
        
#         # Create temporary directory
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Save uploaded file
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(temp_dir, filename)
#             file.save(file_path)
            
#             logger.info(f"Processing file: {filename}")
            
#             # Extract MPNs
#             df_with_mpns = extract_mpn_from_file(file_path)
            
#             # Convert to CSV string
#             csv_buffer = io.StringIO()
#             df_with_mpns.to_csv(csv_buffer, index=False)
#             csv_content = csv_buffer.getvalue().encode('utf-8')
            
#             # Upload to fitment API and get ZIP
#             zip_content = upload_to_fitment_api(csv_content)
            
#             # Transform the ZIP content to expected CSV format
#             transformed_df = transform_fitment_data_to_expected_format(zip_content)
            
#             # Convert transformed DataFrame to CSV
#             output_csv = io.StringIO()
#             transformed_df.to_csv(output_csv, index=False)
#             output_csv.seek(0)
            
#             # Generate output filename
#             base_name = os.path.splitext(filename)[0]
#             output_filename = f"{base_name}_fitment_results.csv"
            
#             # Create BytesIO object for file response
#             csv_bytes = io.BytesIO(output_csv.getvalue().encode('utf-8'))
            
#             return send_file(
#                 csv_bytes,
#                 as_attachment=True,
#                 download_name=output_filename,
#                 mimetype='text/csv'
#             )
            
#     except ValueError as e:
#         logger.error(f"Validation error: {e}")
#         return jsonify({'error': str(e)}), 400
#     except Exception as e:
#         logger.error(f"Processing error: {e}")
#         return jsonify({'error': f'Processing failed: {str(e)}'}), 500

# @app.route('/', methods=['GET'])
# def index():
#     """Root endpoint with API documentation"""
#     return jsonify({
#         'message': 'MPN Fitment API',
#         'version': '1.0',
#         'endpoints': {
#             'POST /process-mpn': 'Upload CSV/Excel file to extract MPNs and get transformed fitment data as CSV',
#             'GET /health': 'Health check endpoint',
#             'GET /': 'This documentation'
#         },
#         'usage': {
#             'description': 'Upload a CSV or Excel file with automotive part titles',
#             'required_columns': ['title' or 'product_title'],
#             'response': 'CSV file with Action, ItemID, Relationship, RelationshipDetails columns'
#         }
#     })

# @app.errorhandler(413)
# def too_large(e):
#     return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

# @app.errorhandler(500)
# def internal_error(e):
#     return jsonify({'error': 'Internal server error'}), 500

# if __name__ == '__main__':
#     # Load model at startup
#     if load_model():
#         logger.info("Starting Flask server...")
#         app.run(host='0.0.0.0', port=5000, debug=False)
#     else:
#         logger.error("Failed to load model. Exiting...")
#         exit(1)



import os
import torch
import zipfile
import requests
import tempfile
import shutil
import io
import pandas as pd
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import re
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------- CONFIG ----------
MODEL_PATH = "fine_tuned_model_MPN8"  # Update this path
EMAIL = "user@example.com"       # Replace with your email
PASSWORD = "securePassword123"   # Replace with your password
LOGIN_URL = 'https://towbarinstructions.co.uk/api/login'
UPLOAD_URL = 'https://towbarinstructions.co.uk/api/uploadFileDownloadMvl/article/upload'

# Flask config
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Global model variables
tokenizer = None
model = None
device = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the ML model once at startup"""
    global tokenizer, model, device
    try:
        logger.info("Loading T5 model...")
        tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
        # model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, from_safetensors=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def detect_title_column(df):
    """Detect the title column in the dataframe"""
    for col in df.columns:
        if col.lower().strip() in ['title', 'product title', 'product_title']:
            return col
    raise ValueError("No title or product_title column found!")

def predict_batch(titles):
    """Predict MPNs for a batch of titles"""
    batch_size = 64 if torch.cuda.is_available() else 4
    predictions = []
    
    for i in tqdm(range(0, len(titles), batch_size), desc="Extracting MPNs"):
        batch = titles[i:i + batch_size]
        prompts = [f"extract only MPN and OEM numbers not a model no. from this automotive part title: {t}" for t in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_new_tokens=20, num_beams=4, early_stopping=True)
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
    
    return predictions

def clean_mpns(mpn_string):
    """Clean and format MPN strings"""
    if pd.isna(mpn_string) or not mpn_string:
        return ""
    mpn_string = re.sub(r'[\/;]', ',', mpn_string)
    mpns = mpn_string.split(',')
    return '|'.join([m.strip() for m in mpns if len(m.strip()) > 4 and not m.strip().isalpha() and '$' not in m])

def explode_mpns(df):
    """Explode MPN column to separate rows"""
    if "MPN" not in df.columns:
        return df
    return df.assign(MPN=df["MPN"].str.split("|")).explode("MPN").reset_index(drop=True)

def detect_itemid_column(df):
    """Detect the ItemID column in the dataframe"""
    for col in df.columns:
        if col.lower().strip() in ['itemid', 'item_id', 'item id', 'id']:
            return col
    return None

def extract_mpn_from_file(file_path):
    """Extract MPNs from uploaded file"""
    try:
        # Read file based on extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Detect title column
        title_col = detect_title_column(df)
        
        # Detect ItemID column (optional)
        itemid_col = detect_itemid_column(df)
        if itemid_col:
            logger.info(f"Found ItemID column: {itemid_col}")
        
        # Predict MPNs
        logger.info("Starting MPN prediction...")
        df['predicted_mpn'] = predict_batch(df[title_col].astype(str).tolist())
        df['MPN'] = df['predicted_mpn'].apply(clean_mpns)
        
        # Explode MPNs
        df = explode_mpns(df)
        
        return df
    except Exception as e:
        logger.error(f"Error extracting MPNs: {e}")
        raise

def upload_to_fitment_api(csv_content):
    """Upload CSV content to fitment API and get ZIP response"""
    try:
        # Login
        logger.info("Logging in to Fitment API...")
        response = requests.post(LOGIN_URL, json={"email": EMAIL, "password": PASSWORD})
        if response.status_code != 200:
            raise Exception(f"Login failed: {response.text}")
        
        token = response.json().get("token")
        
        # Upload file
        logger.info("Uploading file to Fitment API...")
        files = {'file': ('data.csv', csv_content, 'text/csv')}
        headers = {"accept": "application/zip", "Authorization": token}
        r = requests.post(UPLOAD_URL, headers=headers, files=files)
        
        if r.status_code == 200 and 'zip' in r.headers.get('Content-Type', ''):
            logger.info("ZIP received from API")
            return r.content
        else:
            raise Exception(f"Failed to get fitment ZIP: {r.status_code} {r.text}")
            
    except Exception as e:
        logger.error(f"Error uploading to fitment API: {e}")
        raise

def transform_fitment_data_to_expected_format(zip_content, original_itemids=None):
    """Transform the fitment data from ZIP to expected CSV format"""
    try:
        logger.info("Transforming fitment data to expected format...")
        
        # Extract CSV from ZIP
        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_file:
            csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise Exception("No CSV file found in ZIP")
            
            # Read the first CSV file
            with zip_file.open(csv_files[0]) as csv_file:
                df = pd.read_csv(csv_file)
        
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Original columns: {df.columns.tolist()}")
        
        # Group by MPN to handle multiple compatibilities per item
        transformed_rows = []
        
        # Group data by MPN
        grouped = df.groupby('mpn')
        
        for mpn, group in grouped:
            # Skip if MPN is invalid
            if pd.isna(mpn) or str(mpn) == 'nan':
                continue
            
            # Determine ItemID to use
            item_id = str(mpn)  # Default to MPN
            
            # If original ItemIDs were provided, try to find matching one
            if original_itemids and len(original_itemids) > 0:
                # Use the first available ItemID from the original file
                # In practice, you might want to implement better logic to match
                # MPN back to original ItemID if there's a relationship
                item_id = str(original_itemids[0])
            
            # Add the first row with Revise action and ItemID
            first_row = {
                'Action': 'Revise',
                'ItemID': item_id,
                'Relationship': '',
                'RelationshipDetails': ''
            }
            transformed_rows.append(first_row)
            
            # Add compatibility rows for each record in the group
            for _, row in group.iterrows():
                make = str(row.get('make', ''))
                model = str(row.get('model', ''))
                variant = str(row.get('variant', ''))
                engine = str(row.get('engine', ''))
                body_style = str(row.get('body_style', ''))
                type_field = str(row.get('type', ''))
                year = str(row.get('year', ''))
                
                # Create relationship details string
                relationship_details = []
                if make != 'nan' and make: relationship_details.append(f"Make={make}")
                if model != 'nan' and model: relationship_details.append(f"Model={model}")
                if variant != 'nan' and variant: relationship_details.append(f"Variant={variant}")
                if body_style != 'nan' and body_style: relationship_details.append(f"BodyStyle={body_style}")
                if type_field != 'nan' and type_field: relationship_details.append(f"Type={type_field}")
                if engine != 'nan' and engine: relationship_details.append(f"Engine={engine}")
                if year != 'nan' and year: relationship_details.append(f"Cars Year={year}")
                
                relationship_details_str = "|".join(relationship_details)
                
                # Create compatibility row
                compatibility_row = {
                    'Action': 'Add',
                    'ItemID': '',
                    'Relationship': 'Compatibility',
                    'RelationshipDetails': f"Car {relationship_details_str}"
                }
                
                transformed_rows.append(compatibility_row)
        
        # Create DataFrame from transformed rows
        transformed_df = pd.DataFrame(transformed_rows)
        
        logger.info(f"Transformed data shape: {transformed_df.shape}")
        logger.info(f"Sample transformed data:\n{transformed_df.head()}")
        
        return transformed_df
        
    except Exception as e:
        logger.error(f"Error transforming fitment data: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/process-mpn', methods=['POST'])
def process_mpn():
    """Main endpoint to process CSV and return transformed CSV"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV, XLS, XLSX allowed'}), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            
            logger.info(f"Processing file: {filename}")
            
            # Extract MPNs
            df_with_mpns = extract_mpn_from_file(file_path)
            
            # Extract original ItemIDs if they exist
            original_itemids = []
            itemid_col = detect_itemid_column(df_with_mpns)
            if itemid_col:
                original_itemids = df_with_mpns[itemid_col].dropna().unique().tolist()
                logger.info(f"Found {len(original_itemids)} original ItemIDs")
            
            # Convert to CSV string
            csv_buffer = io.StringIO()
            df_with_mpns.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue().encode('utf-8')
            
            # Upload to fitment API and get ZIP
            zip_content = upload_to_fitment_api(csv_content)
            
            # Transform the ZIP content to expected CSV format
            transformed_df = transform_fitment_data_to_expected_format(zip_content, original_itemids)
            
            # Convert transformed DataFrame to CSV
            output_csv = io.StringIO()
            transformed_df.to_csv(output_csv, index=False)
            output_csv.seek(0)
            
            # Generate output filename
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_fitment_results.csv"
            
            # Create BytesIO object for file response
            csv_bytes = io.BytesIO(output_csv.getvalue().encode('utf-8'))
            
            return send_file(
                csv_bytes,
                as_attachment=True,
                download_name=output_filename,
                mimetype='text/csv'
            )
            
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation"""
    return jsonify({
        'message': 'MPN Fitment API',
        'version': '1.0',
        'endpoints': {
            'POST /process-mpn': 'Upload CSV/Excel file to extract MPNs and get transformed fitment data as CSV',
            'GET /health': 'Health check endpoint',
            'GET /': 'This documentation'
        },
        'usage': {
            'description': 'Upload a CSV or Excel file with automotive part titles',
            'required_columns': ['title' or 'product_title'],
            'response': 'CSV file with Action, ItemID, Relationship, RelationshipDetails columns'
        }
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model at startup
    if load_model():
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Failed to load model. Exiting...")
        exit(1)