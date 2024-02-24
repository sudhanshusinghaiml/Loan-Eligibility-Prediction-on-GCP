from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'csvfile' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['csvfile']
    if file.filename == '':
        return 'No file selected', 400
    
    if file:
        # Save the uploaded CSV file
        file.save(file.filename)
        
        # Process the CSV file
        df = pd.read_csv(file.filename)
        # Modify the data as needed
        # For example, let's add a new column with double values of an existing column
        df['new_column'] = df['old_column'] * 2
        
        # Save the modified data to a new CSV file
        output_file = 'modified_data.csv'
        df.to_csv(output_file, index=False)
        
        # Render the HTML template with the modified data
        return render_template('display.html', data=df.to_html())
    else:
        return 'File upload failed', 400

if __name__ == '__main__':
    app.run(debug=True)
