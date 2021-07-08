# importing libraries

import pandas as pd
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc 
import dash_bootstrap_components as dbc
import webbrowser
import plotly.express as px
from dash.dependencies import Output,Input,State
import pickle
import re
from wordcloud import WordCloud,STOPWORDS
from PIL import Image


# from sklearn.feature_extraction.text import TfidfVectorizer

# declaring global Variables
scrappedReviews= None
project_name= None
app=dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)

# defining myfunctions
def load_model():
    global scrappedReviews
    scrappedReviews=pd.read_csv('scrappedReviews.csv')
    
    global recreated_model
    
    file = open("pickle_model.pkl",'rb')
    recreated_model = pickle.load(file)
    
    global vectorizer
    vectorizer=pickle.load(open('vectorizer.pkl','rb'))
    
def scrappedReview_prediction():
    global scrappedReviews
    global recreated_model
    global vectorizer
    
    vectorized_reviews=vectorizer.transform(scrappedReviews['reviews'])    
    pred=recreated_model.predict(vectorized_reviews)
    count_0=0
    count_1=0
    
    for i in range(len(pred)):
        if pred[i]==0:
            count_0=count_0 + 1
        else:
            count_1=count_1 + 1
    
    return [count_0,count_1]


    
def create_app_ui():
    global project_name
    global scrappedReviews
    global app
    main_layout=html.Div(
        [
    dbc.Container(
        [
            dbc.Jumbotron(
                [
                    html.H2(id='sec-1',children='Piechart of Reviews'),
                    dcc.Graph(
                        id='pie_chart',
                        animate=True,
                        figure=pie_chart(scrappedReview_prediction()),
                        style={'margin-bottom':'30px'})
                    ],
                className='text-center',
                style={
                    'margin-bottom':'50px'}
                ),
            dbc.Jumbotron(
                [
                    html.H2(id='sec-2',children='WordCloud-scrappedReviews',
                            style={'color':'black',
                                   'font-weight': 'bold',
                                   'padding':'10'}),

                    html.Img(
                         id='word_cloud',
                         height=400,
                         width=600,
                         src=Word_cloud(),
                         title='scrappedReviews Word cloud',
                         style={'margin-bottom':'30px',
                                })

                    ],
                className='text-center',
                style={
                    'margin-bottom':'50px',
                    'background-color':'white'}
                ),
            
            dbc.Jumbotron( 
                [
                    html.H1(id='main_title',children='Sentiment Analysis with Insights'),
                
                    dcc.Dropdown(
                        id='dropdown_review',
                        placeholder='Select Review.....',
                        clearable = True,
                        
                        style={'width':'100%','margin-bottom':'30px','display':True},
                        className='text-left',
                        options=[{'label':review1,'value':review1} for review1 in scrappedReviews.sample(1000)['reviews']]
                        ),
                    dbc.Button(
                        id='button_dropdown',
                        children='Click to check',
                        size='medium',
                        color='dark',
                        n_clicks=0,
                        style={
                         'width':'200px',
                         'textAlign':'center',
                         'margin-bottom':'30px'},
                        ),
                    dbc.Alert(
                        id='response_dropdown',
                        children='This will show the result of dropdown', 
                        color= None,
                        style={'textAlign':'center'}
                        )
                    ],
                className = 'text-center',
                style={
                    'margin-bottom':'50px'}
                ),
            dbc.Jumbotron( 
                [
                dcc.Textarea(
                    id='textarea_review',
                    placeholder='Enter the reviews here....',
                    style={ 'width': '100%','height':'200px' }
                    ),
                dbc.Button(
                    id='button_textarea',
                    children='Submit',
                    color= 'dark',
                    size= 'Medium',
                    n_clicks=0,
                    className="mt-2 mb-3",
                    style={
                        'width':'100px',
                        'textAlign':'center',
                        'margin-bottom':'30px'
                        
                        }
                    ),
                dbc.Alert(
                    id='response_textarea',
                    children='This will show the result of textarea',
                    color= None,
                    style={'textAlign':'center'}
                    )
            ],
            className = 'text-center',
            style={
                    'background-color':'light',
                    'margin-bottom':'50px'}
        )
        ],
       className='mt-4',
       
       style={
            'background-color':'black',
            'width':'80%'
            }
           )
    ],
        style={
            'background-color':'white'}
       )
    return main_layout

def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')
    
def checkReview(reviewText):
    global recreated_model
    global vectorizer
    vect_review=vectorizer.transform([reviewText])
    pred=recreated_model.predict(vect_review)
    return pred    

def pie_chart(positivity):
    global scrappedReviews

    piechart=px.pie(values=positivity,
                    names=['negative','positive'],
                    title='Reviews Classification',
                    color=['negative','positive'],
                    # color_discrete_sequence=px.colors.sequential.,
                    color_discrete_map={'negative':'red',
                                        'positive':'green'}).update_traces(textposition='inside',
                                                                           textinfo='percent+value+label',
                                                                           pull=[0.2,0],
                                                                           textfont_family='Times New Roman',
                                                                           insidetextfont_family='Open Sans')
    return piechart


def Word_cloud():
    global scrappedReviews
    global app
    word_list=[]
    for review_ in scrappedReviews['reviews']:
        words=re.sub('[^a-zA-Z]',' ',review_)
        words=words.lower()
        words=words.split()
        for i in words:
            word_list.append(i)
    all_reviews=" ".join(word_list)    
    stopwords=set(STOPWORDS)
    #mask=np.array(Image.open('upvote.png'))
    wordcloud=WordCloud(width=800,height=600,background_color='white',stopwords=stopwords,max_words=100,collocations=False,colormap='rainbow')
    wordcloud=wordcloud.generate(all_reviews)
    wordcloud.to_file('Review_wordcloud.png')
    
    return (Image.open('Review_wordcloud.png'))


@app.callback(
    Output('response_textarea','children'),
    Output('response_textarea', 'color'),
    [
    Input('button_textarea','n_clicks')
     ],
    [
    State('textarea_review','value')
     ]
    )
def update_app_ui(n_clicks,textarea_review):
    print('DataType :',str(type(textarea_review)))
    print('Value:',str(textarea_review))
    if(n_clicks>0):
        prediction=checkReview(textarea_review)
    
        if prediction==0:
            result1= 'Negative'
            color1= 'danger'
        elif prediction==1:
            result1= 'Positive'
            color1= 'success'
        else:
            result1= 'Unknown'
            color1= 'primary'
        return result1,color1
    else:
        return ('No reviews selected','dark')

@app.callback(
    Output('response_dropdown','children'),
    Output('response_dropdown','color'),
    [
    Input('button_dropdown','n_clicks')
     ],
    [
    State('dropdown_review','value')
     ]
    )
def update_app_ui_2(n_clicks,dropdown_review):
    print('DataType :',str(type(dropdown_review)))
    print('Value:',str(dropdown_review))
    
    if(n_clicks>0):
        prediction=checkReview(dropdown_review)
    
        if prediction==0:
            result2= 'Negative'
            color2= 'danger'
        elif prediction==1:
            result2= 'Positive'
            color2= 'success'
        else:
            result2= 'Unknown'
            color2= 'primary'
        return result2,color2   
    else:
        return ('No review to show result','dark')

#definining main function which controls the flow of application
def main():
    global project_name
    global scrappedReviews
    global app
    global recreated_model
    global vectorizer
    
    project_name = "Sentiment Analysis with Insights"

    print('Start of my project')
    load_model()
    open_browser()
    print('Wait Project is Loading....')
    #print('My Project name is:',project_name)
    #print('My project Data:',scrappedReviews.sample(5))
    app.title= project_name
    app.layout= create_app_ui()
    #,dev_tools_serve_dev_bundles=True,dev_tools_prune_errors=True.,
    app.run_server(host='0.0.0.0', port=8050,debug=True,dev_tools_hot_reload=True,dev_tools_ui=True,dev_tools_props_check=True,use_reloader=False)  
    
    print('End of my project')
    scrappedReviews= None
    app=None
    project_name=None
    recreated_model=None
    vectorizer=None

# calling the main function
if __name__=='__main__':
    main()

