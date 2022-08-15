library("fastDummies")
library(MASS)
library(leaps)
library(caret)
library(dplyr)
#library(ggplot2)
#library(reshape2)
library(MLmetrics)
library(olsrr)
library(corrplot)


#Pre-processing (preliminary)
Insurance_Data <- read.csv('insurance.csv')

Insurance_Pre <- dummy_cols(Insurance_Data, select_columns = c('sex','smoker','region')) 
                              #remove_first_dummy = TRUE)

Insurance_Final <- Insurance_Pre[,c('age','sex_female','sex_male','bmi','children','smoker_no','smoker_yes',
                                   'region_northeast','region_northwest','region_southeast',
                                   'region_southwest','charges')]

#Correlation plot of numerical variables
corrplot(cor(Insurance_Final[c(1, 4, 5, 12)]))

#Missing value check
which(is.na(Insurance_Final))

#Final CSV for processed data
write.csv(Insurance_Final,"C:/Users/sraba/OneDrive/Documents/R//Processed_Insurance.csv", row.names = FALSE)

#Test-Train data split
dt = sort(sample(nrow(Insurance_Final), nrow(Insurance_Final)*.73))
Insurance_Final_train_73pc<-Insurance_Final[dt,]
Insurance_Final_test_27pc<-Insurance_Final[-dt,]

#Test-Train CSV files
write.csv(Insurance_Final_train_73pc,"C:/Users/sraba/OneDrive/Documents/R//Processed_Insurance_train_73pc.csv", row.names = FALSE)
write.csv(Insurance_Final_test_27pc,"C:/Users/sraba/OneDrive/Documents/R//Processed_Insurance_test_27pc.csv", row.names = FALSE)

#Boxplot for outliers
par(mfrow=c(2,2))
boxplot(Insurance_Final_train_73pc$age, col=2, xlab='AGE')
boxplot(Insurance_Final_train_73pc$bmi, col=3, xlab='BMI')
boxplot(Insurance_Final_train_73pc$children, col=4, xlab='CHILDREN')
boxplot(Insurance_Final_train_73pc$charges, col=5, xlab='CHARGES')

#Histogram with line and density line
par(mfrow=c(1,2))

hist(Insurance_Final_train_73pc$charges, prob=TRUE, col=8)
lines(density(Insurance_Final_train_73pc$charges),col='red',lwd=4)

age_grid <- seq(min(Insurance_Final_train_73pc$charges), max(Insurance_Final_train_73pc$charges))
func <- dnorm(age_grid, mean = mean(Insurance_Final_train_73pc$charges), sd = sd(Insurance_Final_train_73pc$charges))
hist(Insurance_Final_train_73pc$charges, prob=TRUE, ylim = c(0, max(func)), col=15)
lines(age_grid, func, col='red',lwd=4)


#Model fitting
null <- lm(charges~1,data=Insurance_Final_train_73pc)
model_wo_transform <- lm(charges~.,data=Insurance_Final_train_73pc)


#Boxcox transformation and model fitting
bc <- boxcox(charges ~ .,data=Insurance_Final_train_73pc)
(lambda <- bc$x[which.max(bc$y)])

model_w_transform <- lm(((charges^lambda-1)/lambda) ~ .,data=Insurance_Final_train_73pc)

#Error in prediction (MAPE)
predicted_wo_transform <- data.frame(predict(model_wo_transform,Insurance_Final_test_27pc, se.fit = TRUE))
predicted_w_transform <- data.frame(predict(model_w_transform,Insurance_Final_test_27pc, se.fit = TRUE))
mean(abs((Insurance_Final_test_27pc$charges-predicted_wo_transform$fit)/Insurance_Final_test_27pc$charges)) * 100
mean(abs((((Insurance_Final_test_27pc$charges^lambda-1)/lambda-predicted_w_transform$fit))/((Insurance_Final_test_27pc$charges^lambda-1)/lambda))) * 100

#MAPE through package to confirm
MAPE(predicted_w_transform$fit, (Insurance_Final_test_27pc$charges^lambda-1)/lambda)
MAPE(predicted_wo_transform$fit,Insurance_Final_test_27pc$charges)

#The polynomila-regression fit
poly_model <- lm(charges ~ bmi+children+smoker_yes+poly(age,2)+poly(bmi,2)+poly(children,2)+
                           bmi:smoker_yes+children:smoker_yes, data=Insurance_Final_train_73pc)
predicted_poly <- data.frame(predict(poly_model,Insurance_Final_test_27pc, se.fit = TRUE))
MAPE(predicted_poly$fit,Insurance_Final_test_27pc$charges)

#R-square
summary(model_wo_transform)
summary(model_w_transform)
summary(poly_model)


#Multi-collinearity check
correlation_matrix=round(cor(Insurance_Final_train_73pc),5)

#Variable Selection
VS_best_subset <- regsubsets(charges~.,data=Insurance_Final_train_73pc,nbest=1)
info <- summary(VS_best_subset)
res <- data.frame(cbind(info$which, round(cbind(rsq=info$rsq, adjr2=info$adjr2,
                                                bic=info$bic, rss=info$rss), 3)))

stepAIC(model_w_transform, scope=list(lower=null, upper=model_w_transform), 
        data=Insurance_Final_train_73pc, direction='backward')

stepAIC(null, scope=list(lower=null, upper=model_w_transform), 
        data=Insurance_Final_train_73pc, direction='forward')

stepAIC(null, scope=list(lower=null, upper=model_w_transform), 
        data=Insurance_Final_train_73pc, direction='both')

#New model fit after variable selection
new_model_w_transform <-lm(((charges^lambda-1)/lambda) ~ age+sex_female+bmi
                                                        +children+smoker_no
                                                        +region_northeast
                                                        +region_northwest
                                                        +region_southeast,data=Insurance_Final_train_73pc)
summary(new_model_w_transform)

#Prediction
predicted <- data.frame(predict(new_model_w_transform,Insurance_Final_test_27pc, se.fit = TRUE))

MAPE(predicted$fit,(Insurance_Final_test_27pc$charges^lambda-1)/lambda)

sqrt(mean(((Insurance_Final_test_27pc$charges^lambda-1)/lambda - predicted$fit)^2))

#Cross-validation
Insurance_Final_train_73pc_mutated <- mutate(Insurance_Final_train_73pc,new_charges=(charges^lambda-1)/lambda)
train_control <- trainControl(method = "LOOCV")
model <- train(new_charges ~ age+sex_female+bmi+children+smoker_no+region_northeast
               +region_northwest+region_southeast, data = Insurance_Final_train_73pc_mutated,
               method = "lm",
               trControl = train_control)
print(model)


#Residual analysis
par(mfrow=c(1,3))
ols_plot_resid_qq(new_model_w_transform)
ols_plot_resid_fit(new_model_w_transform)
ols_plot_resid_hist(new_model_w_transform)

ols_test_normality(new_model_w_transform)

ols_plot_resid_qq(poly_model)
ols_plot_resid_fit(poly_model)
ols_plot_resid_hist(poly_model)

ols_test_normality(poly_model)

#AIC of polynomial model and transformed linear model
extractAIC(new_model_w_transform)
extractAIC(poly_model)