from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import pandas as pd
import scipy.stats as stats
df=pd.read_csv("train.csv")
dat={'OverallQual':df['OverallQual'],  
     'GrLivArea':df['GrLivArea'],
     'GarageArea':df['GarageArea'],
     'FullBath':df['BsmtFullBath']+df['FullBath']+(df['BsmtHalfBath']+df['HalfBath']),
     'SalePrice':df['SalePrice']}
dat=pd.DataFrame(dat); df=dat
df.drop_duplicates(inplace=True)
z_scores=stats.zscore(df); threshold=3
df=df[(z_scores<=threshold).all(axis=1)]
x=df.drop('SalePrice',axis=1)
y=df['SalePrice']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
predict_output=model.predict(x_test)
print("mean_absolute_error:",mean_absolute_error(y_test,predict_output))
print("mean_squared_error:",mean_squared_error(y_test,predict_output))
print("r2_score:",r2_score(y_test,predict_output)) 
def predict_house_price(model):
    try:
        overall_qual = float(input("Enter Overall Quality (1-10): ").strip())
        gr_liv_area = float(input("Enter Ground Living Area (sq ft): ").strip())
        garage_area = float(input("Enter Garage Area (sq ft): ").strip())
        full_bath = float(input("Enter Full Bath (combined): ").strip())
        user_input = pd.DataFrame([[overall_qual, gr_liv_area, garage_area, full_bath]], 
                                  columns=['OverallQual', 'GrLivArea', 'GarageArea', 'FullBath'])
        prediction = model.predict(user_input)
        print(f"The estimated house price is: ${prediction[0]:,.2f}")
    except ValueError:
        print("Invalid input. Please enter numeric values for all fields.")
    except Exception as e:
        print(f"An error occurred: {e}")
choice = 1
while choice:
    predict_house_price(model)
    choice = int(input("Enter 1 to Continue or 0 to Exit:"))
    print("\n")