# Chapter 2

## ML project checklist :white_check_mark:

- [ ] Frame the problem and look at the big picture
- [ ] Get the data
- [ ] Explore data to gain insights
- [ ] Prepare data to better expose the underlying data patterns to ML algos
- [ ] Explore different models and shortlist the best one
- [ ] Fine tune your models and combine then in a solution
- [ ] Present the solution
- [ ] Launch, monitor and maintain

## ML Project :books:

Predicting housing prices using the 1990 california census dataset. 

### Framing the problem

The objective of the project is that the prediction from our model is going to be fed to another ML model to predict whether the firm should invest in that area or not.

It is a supervised learning regression problem that doesn't have very reactive data (batch). It is a **multiple regression** problem since multiple features will be used to make prediction. Moreover, it's a **univariate** **regression** since we are only predicting one thing. 

Performance measure would be **RMSE** 

> #### ASIDE :nerd_face:
>
> RMSE and MAE are ways to measure the distance between two vectors. Various distance measures called ***norms*** are possible.
>
> * RMSE corresponds to the **Euclidean** norm or the **l~2~** norm
>
> * MAE is called **Manhattan** norm or the **l~1~** norm. It is called so because it measures the distance between two points in a city if you can travel only in orthogonal city blocks
>
> * In general, **l~k~** norm is defined as - 
>   $$
>   ||v||_k = (|v_0|^k + |v_1|^k....|v_n|^k)^{(1/k)}
>   $$
>
> * **The higher the norm index, the more it focuses on large values and neglects small ones. In that sense RMSE is more sensitive to outliers than MAE** 
>
> * For *Bell shaped* curves, outliers are very less and hence RMSE performs better.

