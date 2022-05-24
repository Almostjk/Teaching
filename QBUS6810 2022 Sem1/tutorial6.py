import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected = True)

import numpy as np

def overfitting():

    X_train = train[['Limit']].to_numpy()

    x = np.linspace(X_train.min(), X_train.max(), 500).reshape((-1,1))
    y_fitted=np.zeros((len(x),50))
    
    mse = np.zeros(50)
    for k in range(1,51):
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        y_fitted[:,k-1] = knn.predict(x)
        
        mse[k - 1] = np.mean((knn.predict(X_train) - y_train) ** 2)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=train['Limit'], y=train['Balance'], mode='markers',
            marker_color=colours[0],
            name='Data',
            showlegend=True,
    ))

    for k in range(50):
        fig.add_trace(go.Scatter(x=np.ravel(x), y=y_fitted[:,k], mode='lines',
                visible=False,
                showlegend=False,
                line_color='#444',
        ))

    fig.data[10].visible = True

    # Slider
    steps = []
    for i in range(len(fig.data)):
        if i % 3 == 0:
            label=str(i)
        else:
            label=''
        step = dict(
            method="update",
            args=[{"visible": [False] * (len(fig.data))},
                  {"title": "Number of neighbours: " + str(i) + f"; Train MSE: {round(mse[i - 1], 2)}"},
                 ],
            label=str(i),
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][0] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=9,
        currentvalue={"prefix": "k="},
        pad={"t": 25},
        steps=steps[1:],
        minorticklen=0,
        ticklen=0,
    )]

    fig.update_layout(
        sliders=sliders
    )

    fig.update_layout(xaxis_title='Limit', xaxis_showticklabels=False, 
                     yaxis_title='Balance', yaxis_showticklabels=False)
    fig.update_layout(template='simple_white', width=700, height=500, margin=dict(l=50, r=50, b=15, t=50, pad=4))

    fig.show()


def biasvariance():
    predictors = ['Limit', 'Income']
    X_train = train[predictors]
    X_valid = valid[predictors]

    n_neighbours=np.arange(1, 51)

    test_rmse = []
    for k in n_neighbours: 
        knn = KNeighborsRegressor(n_neighbors=k, metric='mahalanobis', metric_params={'V': X_train.cov()})
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_valid) 
        test_rmse.append(np.sqrt(mean_squared_error(y_valid, y_pred)))


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=n_neighbours, y= test_rmse, mode='lines',
            line=dict(color=colours[0], width=2),
            name='',
            connectgaps=True,
            hovertemplate =
            '<b>k</b> = %{x}'+
            '<br>Validation RMSE</b> = %{y:.1f}<br>',
            showlegend=False
    ))
          
    fig.update_layout(xaxis_title='Number of neighbours', yaxis_title='Test RMSE', yaxis_showticklabels=False)
    fig.update_layout(template='plotly_white', width=700, height=450, margin=dict(l=50, r=50, b=15, t=15, pad=4))

    fig.show()


def curse():
    # Ordering the predictors according to their correlation with the response
    # Tip: you can often make sense of code by printing partial outputs
    predictors = list(train.corr()[response].sort_values(ascending=False).index[1:])

    p = len(predictors)
    y_pred_knn = np.zeros((len(valid),p))
    y_pred_ols = np.zeros((len(valid),p))

    X_train = train[predictors]
    X_valid = valid[predictors]

    for i in range(1,p+1): 
        # kNN predictions
        knn = KNeighborsRegressor(n_neighbors=3, metric='mahalanobis', 
                                  metric_params={'V': X_train.iloc[:,:i].cov()})
        knn.fit(X_train.iloc[:,:i], y_train)
        y_pred_knn[:, i-1] = knn.predict(X_valid.iloc[:,:i]) 
        
        # Linear 
        ols = LinearRegression()
        ols.fit(X_train.iloc[:,:i], y_train)
        y_pred_ols[:, i-1] = ols.predict(X_valid.iloc[:,:i]) 

    y_pred_ols[y_pred_ols<0] =0 
                                  
    # Initialise table
    columns=['RMSE (kNN)', 'RMSE (linear)']
    rows=1+np.arange(p)
    results = pd.DataFrame(0.0, columns=columns, index=rows)
    results.index.name = 'p'

    # Computer test predictions and metrics
    for i in range(p):
        results.iloc[i, 0] = np.sqrt(mean_squared_error(y_valid, y_pred_knn[:,i]))  
        results.iloc[i, 1] = np.sqrt(mean_squared_error(y_valid, y_pred_ols[:,i]))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=results.index, y=results.iloc[:,0], mode='lines',
            line=dict(color=colours[0], width=2),
            name='kNN',
            connectgaps=True,
             hoverinfo = 'y',
            hovertemplate =
            'kNN' +
            '<br>p = %{x}'+
            '<br>Validation RMSE = %{y:.1f}<br><extra></extra>',
            showlegend=True
    ))

    fig.add_trace(go.Scatter(x=results.index, y=results.iloc[:,1], mode='lines',
            line=dict(color=colours[1], width=2),
            name='Linear regression',
            connectgaps=True,
            hovertemplate =
            'Linear regression' +
            '<br>p = %{x}'+
            '<br>Validation RMSE = %{y:.1f}<br><extra></extra>',
            showlegend=True
    ))

    fig.update_layout(xaxis_title='Number of predictors', yaxis_title='Validation RMSE', yaxis_showticklabels=False)
    fig.update_layout(template='plotly_white', width=800, height=450, margin=dict(l=50, r=50, b=15, t=15, pad=4))

    fig.show()

    return results

def kfold():
    # Compute the CV error for k=1 to k=50
    ks=np.arange(1, 51)
    cv_errors = []
    for k in ks: 
        model = KNeighborsRegressor(n_neighbors= k, metric='mahalanobis', metric_params={'V': X_train.cov()}) 
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring = 'neg_mean_squared_error')
        rmse = np.sqrt(-1*np.mean(scores))
        cv_errors.append(rmse)

    # Make figure 
    fig = px.line(x=ks, y=cv_errors, # everything below is detail
                  labels={'x': 'k', 'y': 'RMSE'}, 
                  template='plotly_white',
                  color_discrete_sequence=px.colors.qualitative.T10)

    fig.update_layout(xaxis_title='Number of neighbours', yaxis_title='Cross-validation RMSE', )
    fig.update_layout(width=800, height=500)
    fig.show()


def repeatedkfold():
    # Compute the CV error for k=1 to k=50
    ks=np.arange(1, 51)
    cv_errors = []
    for k in ks: 
        model = KNeighborsRegressor(n_neighbors= k, metric='mahalanobis', metric_params={'V': X_train.cov()}) 
        scores = cross_val_score(model, X_train, y_train, cv=rkf, scoring = 'neg_mean_squared_error')
        rmse = np.sqrt(-1*np.mean(scores))
        cv_errors.append(rmse)

    # Make figure 
    fig = px.line(x=ks, y=cv_errors, 
                  labels={'x': 'k', 'y': 'RMSE'}, 
                  template='plotly_white',
                  color_discrete_sequence=px.colors.qualitative.T10)

    fig.update_layout(xaxis_title='Number of neighbours', yaxis_title='Cross-validation RMSE', )
    fig.update_layout(width=800, height=500)
    fig.show()