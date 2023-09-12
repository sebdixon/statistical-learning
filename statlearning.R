df = read.csv("~/Downloads/diamonds.csv")
library(dplyr) #Data manipulation
library(plyr) #data manipulation
library(ggplot2) #For graph plotting
library(corrplot) #for correlation plot
library(gridExtra) #For multiple plots
library(mlbench) #For feature engineering
library(caret) #For feature engineering


summary(df)

df$X <- NULL
df$cut <- as.factor(diamonds$cut) #Change the cut variable to a factor
df$color <- as.factor(diamonds$color) #Change the color variable to a factor
df$clarity <- as.factor(diamonds$clarity)




boxplot(df$depth, main="Depth")
boxplot(df$table, main="Table")
boxplot(df$price)
df <- subset(df, df$depth > 50 & df$depth < 75)

df <- subset(df, df$table > 47 & df$table < 80)

boxplot(df$x)
df <- subset(df, df$x > 2 & df$x < 11)
boxplot(df$y)
df <- subset(df, df$y < 15)
boxplot(df$z)
df <- subset(df, df$z > 1.2 & df$z < 7.5)


categorical_var <- df[,c(2,3,4)]
continous_var <- df[,-c(2,3,4)]

#Creating data_main for modelling
data_main <- cbind(continous_var,categorical_var) 
par(mfrow=c(1,1)) # to divide the grid area

# Plotting cut vs price
a <- ggplot(data = data_main)+
  geom_bar(aes(x=cut,y=price),stat = "summary", alpha=1,fill="blue")+
  xlab("Cut Type")+
  ylab("Average Price")+
  theme(axis.text=element_text(size=4))+
  ggtitle("Cut vs Price")

# Plotting color vs price
b <- ggplot(data = data_main)+
  geom_bar(aes(x=color,y=price),stat = "summary", alpha=1,fill="red")+
  xlab("Color")+
  ylab("Average Price")+
  theme(axis.text=element_text(size=4))+
  ggtitle("Color vs Price")

# Plotting clarity vs price
c <- ggplot(data = data_main)+
  geom_bar(aes(x=clarity,y=price),stat = "summary", alpha=1,fill="green")+
  xlab("Clarity Type")+
  ylab("Average Price")+
  theme(axis.text=element_text(size=4))+
  ggtitle("Clarity vs Price")

# Plotting carat vs price
d <- ggplot(data = data_main)+
  geom_point(size=0.5,aes(x=carat,y=price),stat = "summary", alpha=1,fill="orange")+
  xlab("Carat")+
  ylab("Price")+   
  theme(axis.text=element_text(size=4))+
  ggtitle("Carat vs Price")

# Plotting depth vs price
e <- ggplot(data = data_main)+
  geom_point(size=0.5,aes(x=depth,y=price),stat = "summary", alpha=1,fill="pink")+
  xlab("Depth")+
  ylab("Price")+   
  theme(axis.text=element_text(size=4))+
  ggtitle("Depth vs Price")

# Plotting table vs price
f <- ggplot(data = data_main)+
  geom_point(size=0.5,aes(x=table,y=price),stat = "summary", alpha=1,fill="purple")+
  xlab("Table")+
  ylab("Price")+   
  theme(axis.text=element_text(size=4))+
  ggtitle("Table vs Price")

# Plotting X vs price
g <- ggplot(data = data_main)+
  geom_point(size=0.5,aes(x=x,y=price),stat = "summary", alpha=1,fill="purple")+
  xlab("X")+
  ylab("Price")+   
  theme(axis.text=element_text(size=4))+
  ggtitle("X vs Price")

# Plotting Y vs price
h <- ggplot(data = data_main)+
  geom_point(size=0.5,aes(x=y,y=price),stat = "summary", alpha=1,fill="purple")+
  xlab("Y")+
  ylab("Price")+   
  theme(axis.text=element_text(size=4))+
  ggtitle("Y vs Price")

# Plotting Z vs price
i <- ggplot(data = data_main)+
  geom_point(size=0.5,aes(x=z,y=price),stat = "summary", alpha=1,fill="purple")+
  xlab("Z")+
  ylab("Price")+   
  theme(axis.text=element_text(size=4))+
  ggtitle("Z vs Price")

# Arrange each plots in grid

grid.arrange(a, b, c,d,e,f,g,h,i, ncol = 3, nrow = 3)

################# FIRST LIENAR MODEL #####################
model1 <- lm(price~., data=data_main)
summary(model1)
residuals1 <- resid(model1)
hist(residuals1)

df$vol = with(df, x*y*z)
df$logprice <- log(df$price)
df$carat <- log(df$carat + 1)

df$carat[is.na(df$Price)]<- median(df$Price,na.rm = TRUE)
df
ix <- sample(53940,40000,replace=FALSE)
df_train<-df[ix,]
df_test<-df[-ix,]

y_train <- df_train$price
x_train <- df_train[, !names(df) %in% c("price","logprice", "table", "X")]
x_train_matrix = makeX(x_train, na.impute=TRUE)

y_test <- df_test$price
x_test <- df_test[, !names(df) %in% c("price","logprice", "table", "X")]
x_test_matrix <- makeX(x_test, na.impute=TRUE)


with (df_train, plot(price, exp(carat)))

glm1.out <- lm(y_train ~., data = x_train)
summary(glm1.out)                 
plot(glm1.out)

y_pred1 = predict(glm1.out, x_test)

MSE <- function(x,y) {
  squared_residuals = y**2 - x**2
  mse <- mean(squared_residuals)
  return(mse)
}

sqrt(MSE(y_pred1, y_test))
#plot(exp(y_pred1), exp(y_test))
glm2 <- cv.glmnet(x_train_matrix, y_train, alpha=1)
bestlam = glm2$lambda.min
plot(glm2) 

glm2_best <- glmnet(x_train_matrix, y_train, alpha=1, lambda=bestlam)
