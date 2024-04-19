from dash import Dash,html,dcc,callback
from dash.dependencies import Input,Output,State
from dash.exceptions import PreventUpdate
from sklearn.cluster import KMeans
import plotly.express as px
import dash_daq as daq
import pandas as pd
import numpy as np
import io
import base64
import Processing 
import pickle


App=Dash()

App.title="Customer Segmentation"


upload_data=dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ],className='updata'),
        style={
            'width': '95%',
            'height': '90%',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderRadius': '10px',
            'textAlign': 'center'
        },
        # Allow multiple files to be uploaded
        multiple=True
    )
  
#spinner
spinner=dcc.Loading(
                    id="load",
                    children=[
            html.Div(id='spin')
        ],
                    type="graph"
                )

#toggle
toggle=daq.ToggleSwitch(
    id="toggle",
    style={'padding-top':'4%',"margin":'2%'}
)




App.layout= html.Div([
    # Side navigation bar
    html.Div([
        html.Div(children=[
            html.H4("Make sure your dataset have following columns:"),
            html.P('Invoice,StockCode,Description,\n Quantity,InvoiceDate,Price,Customer ID,Country,Monetary,Recency',style={"background-color":"red","font-size":"90%","padding":"2%","width":"94%"}),
            html.P("StockCode and Description columns are not mandatory."),
            html.Hr(style={"width":"94%","height":"1px","background-color":"white"}),
            html.Div(
           [html.H5("Silhouette Plot:",style={"background-color":"brown","width":"39%","padding":"2%"}),
            html.Div([
              html.H5("Hide"),
              toggle,
              html.H5('Show'),
            ],className="itog")],className="tog"),
           html.P('The silhouette algorithm is one of the many algorithms to determine the optimal number of clusters for an unsupervised learning technique.',style={"background-color":"blue","font-size":"85%","padding":"2%","width":"94%"}),
           html.Div(html.H4("Chatbot coming soon..."))
            ], style={'height': '100vh'},className='nav-inner')
            
    ], style={'width': '20%', 'display': 'inline-block','position':'fixed'},className="nav"),
    
    # Main content area
    html.Div([
        # Title row
        html.Div([
            html.H1("Customer Segmentation") ,
            html.P('Customer segmentation through K-means clustering is a vital strategy for businesses, enabling them to group customers with similar traits or behaviors. This method facilitates targeted marketing efforts, personalized product development, and efficient resource allocation. By understanding distinct customer segments, businesses can enhance customer retention, mitigate risks, and identify opportunities for expansion. Additionally, segmentation aids in optimizing the customer experience and evaluating the performance of various initiatives. Ultimately, this approach empowers businesses to better understand and serve their customer base, driving growth, profitability, and satisfaction.')
        ], style={'width': '95%', 'display': 'flex', 'justify-content': 'center','flex-direction':'column'},className="title"),
        
        # Content rows
        html.Div([

            # small div for data collection
            html.Div([
                html.Div([upload_data,
                          html.Button('Submit', id='submit-button', n_clicks=0)
                    ], style={'height': '120px'})
            ],style={'width': '95%', 'display': 'inline-block'},id="data-collect"),

           #spinner
           html.Div(children=[spinner]),

           html.Div(id='progress-output'),

            # First box
            html.Div([
                html.Div(id="output",style={'height': '1520px'}),
            ], style={'width': '95%', 'display': 'inline-block'},className="graph-body"),
             # Second box
            html.Div(id='silo')
          ], style={'width': '100%', 'display': 'flex', 'justify-content': 'center','flex-direction':'column'})
    ], style={'width': '80%', 'display': 'inline-block', 'vertical-align': 'top'},className='main')
])


@App.callback(
    Output('progress-output','children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_progress_bar(list_of_contents, list_of_names):
    if list_of_contents is None:
        return ""

    total_files = len(list_of_contents)
    files_uploaded = sum([1 for content in list_of_contents if content is not None])
    percentage_uploaded = int(files_uploaded / total_files * 100) if total_files > 0 else 0

    if percentage_uploaded==100:
     return html.Div([
          html.H5("Uploaded Succefully")
        ],style={"margin-left":"42%","background-color":"red","height":"20px","width":"155px","margin-top":"-4%","padding-left":"2%"})
    else:
      return html.Div([
        html.Progress(value=str(percentage_uploaded), max=100)
        ],style={"margin-left":"42%","margin-top":"-4%"})





# Callback to update the output
@App.callback([Output('output', 'children'),
               Output('spin','children')],
               [Input('submit-button', 'n_clicks')],
              [State('upload-data', 'contents'),
               State('upload-data', 'filename')]
               )
def update_output(n_clicks,list_of_contents, list_of_names):
    if n_clicks == 0:
        raise PreventUpdate
   
    if list_of_contents is not None:
         children = [
            parse_contents(c,n) for c, n in zip(list_of_contents, list_of_names)
         ]
         
         return children,""
     
    return "",""
      
# Function to parse the contents of the uploaded file
def parse_contents(contents, filename,n=0):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assuming the format is CSV
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assuming the format is Excel
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        return 'There was an error processing this file.'

    rfm=Processing.process(df)

    
    plots=html.Div([
          dcc.Graph(figure=px.strip(rfm,y='Recency',x="Customer_label",title='Recency'),style={"height":"33.3%","width":"98%","padding-bottom":"25px","padding-left":"1%"}),
          dcc.Graph(figure=px.strip(rfm,y='Frequency',x="Customer_label",title='Frequency'),style={"height":"33.3%","width":"98%","padding-bottom":"25px","padding-left":"1%"}),
          dcc.Graph(figure=px.strip(rfm,y='Monetary',x="Customer_label",title='Monetary'),style={"height":"33.3%","width":"98%","padding-bottom":"25px","padding-left":"1%"})
          ],style={"height":"1400px"})

    
    return plots
 


# toggle button  
@App.callback(Output("silo",'children'), #Output should be first
              Input("toggle","value")
               )
def show_silo(value):
    if value==True:
        # Data
        with open("Data/processed_df.csv",'rb') as f:
            df=pd.read_csv(f)
        
        img=open("assets/silo.png","rb")
        img_base64 = base64.b64encode(img.read()).decode('ascii')

        return html.Div([html.H3("Silhouette Plot"),
                         html.P("Hint : Select the number of clusters just below the maximum average silhouette_score"),
            html.Img(src='data:image/png;base64,{}'.format(img_base64), style={'width': "100%","height":"20%"})])




if __name__=='__main__':
    App.run(debug=True,dev_tools_hot_reload=False)