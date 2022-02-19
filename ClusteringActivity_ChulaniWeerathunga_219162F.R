#import libraries
library(corrplot)
library(mlbench)
library(dplyr)
library(caret)
library(ggplot2)
library(lattice)

#Read the original data set
wineData = read.csv("wineDataset.csv", header = TRUE)

#View the loaded data set
View (wineData)

#Summary of the original data set
summary(wineData)

#======================== Data cleaning ===============================================

#Exclude the first column (Class labels) from the data set 
wineDataExcludeClass = wineData[, -1]

#Summary of the class label excluded data set
summary(wineDataExcludeClass)

#Plot the data set to visualize any relationships
plot(wineDataExcludeClass)

#Check for name irregularities/ mislabeled variables
names(wineDataExcludeClass)

#Rename the word "NonflavanoidPhenols" to camel case to align with others
names(wineDataExcludeClass)[8] <- "NonFlavanoidPhenols"
names(wineDataExcludeClass)

#Check for the data types of the attributes/features
typeof(wineDataExcludeClass$Alcohol)
typeof(wineDataExcludeClass$MalicAcid)
typeof(wineDataExcludeClass$Ash)
typeof(wineDataExcludeClass$AlcalinityOfAsh)
typeof(wineDataExcludeClass$Magnesium)
typeof(wineDataExcludeClass$TotalPhenols)
typeof(wineDataExcludeClass$Flavanoids)
typeof(wineDataExcludeClass$NonFlavanoidPhenols)
typeof(wineDataExcludeClass$Proanthocyanins)
typeof(wineDataExcludeClass$ColorIntensity)
typeof(wineDataExcludeClass$Hue)
typeof(wineDataExcludeClass$OD280.OD315)
typeof(wineDataExcludeClass$Proline)

#Check for null values in the data set
sum(is.na(wineDataExcludeClass))

#Check for outliers
boxplot(wineDataExcludeClass$Alcohol)

boxplot(wineDataExcludeClass$MalicAcid) # Graph has outliers displayed (3 outliers)
summary(wineDataExcludeClass$MalicAcid)
mallicAcidHist <- hist(wineDataExcludeClass$MalicAcid)
text(mallicAcidHist$mids,mallicAcidHist$counts,labels=mallicAcidHist$counts, adj=c(0.5, -0.5))

boxplot(wineDataExcludeClass$Ash) # Graph has outliers displayed (3 outliers)
summary(wineDataExcludeClass$Ash)
AshHist <- hist(wineDataExcludeClass$Ash)
text(AshHist$mids,AshHist$counts,labels=AshHist$counts, adj=c(0.5, -0.5))

boxplot(wineDataExcludeClass$AlcalinityOfAsh) # Graph has outliers displayed (3 outliers)
summary(wineDataExcludeClass$AlcalinityOfAsh)
AlcalinityOfAshHist <- hist(wineDataExcludeClass$AlcalinityOfAsh)
text(AlcalinityOfAshHist$mids,AlcalinityOfAshHist$counts,labels=AlcalinityOfAshHist$counts, adj=c(0.5, -0.5))

boxplot(wineDataExcludeClass$Magnesium) # Graph has outliers displayed (4 outliers)
summary(wineDataExcludeClass$Magnesium)
MagnesiumHist <- hist(wineDataExcludeClass$Magnesium)
text(MagnesiumHist$mids,MagnesiumHist$counts,labels=MagnesiumHist$counts, adj=c(0.5, -0.5))

boxplot(wineDataExcludeClass$TotalPhenols)
boxplot(wineDataExcludeClass$Flavanoids)
boxplot(wineDataExcludeClass$NonFlavanoidPhenols)

boxplot(wineDataExcludeClass$Proanthocyanins) # Graph has outliers displayed (2 outliers)
summary(wineDataExcludeClass$Proanthocyanins)
ProanthocyaninsHist <- hist(wineDataExcludeClass$Proanthocyanins)
text(ProanthocyaninsHist$mids,ProanthocyaninsHist$counts,labels=ProanthocyaninsHist$counts, adj=c(0.5, -0.5))

boxplot(wineDataExcludeClass$ColorIntensity) # Graph has outliers displayed (3 outliers)
summary(wineDataExcludeClass$ColorIntensity)
ColorIntensityHist <- hist(wineDataExcludeClass$ColorIntensity)
text(ColorIntensityHist$mids,ColorIntensityHist$counts,labels=ColorIntensityHist$counts, adj=c(0.5, -0.5))

boxplot(wineDataExcludeClass$Hue) # Graph has outliers displayed (1 outlier)
summary(wineDataExcludeClass$Hue)
HueHist <- hist(wineDataExcludeClass$Hue)
text(HueHist$mids, HueHist$counts,labels=HueHist$counts, adj=c(0.5, -0.5))

boxplot(wineDataExcludeClass$OD280.OD315)
boxplot(wineDataExcludeClass$Proline)

#Compute Correlation matrix
CorrelationMatrix <- cor(wineDataExcludeClass)
round(CorrelationMatrix, 2)
corrplot(CorrelationMatrix, type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)

#=================================== Feature engineering =====================================

highlyCorrM <- findCorrelation(CorrelationMatrix, cutoff=0.6)
names(wineDataExcludeClass)[highlyCorrM] #Can drop "Flavanoids" and "TotalPhenols"

wineDataAfterDrop = subset(wineDataExcludeClass, select = -c(Flavanoids, TotalPhenols))
summary(wineDataAfterDrop)
plot(wineDataAfterDrop)

#================= Clustering ==================================================

wineDataKMeans = kmeans(wineDataAfterDrop, 3)
wineDataKMeans

plot(wineDataAfterDrop, col=wineDataKMeans$cluster)

#================== Cluster evaluation =========================================

expected_values <- factor(c(wineData$ï..Class))
predicted_values <- factor(c(wineDataKMeans$cluster))

confusionMatrix <- confusionMatrix(data=predicted_values, reference = expected_values)
confusionMatrix

#============= Attempt 2 -  PCA ================================================

#============== Feature engineering ============================================

wineDataPCA = prcomp(wineDataExcludeClass, center = TRUE, scale = TRUE)
summary(wineDataPCA)

variance = wineDataPCA$sdev^2 / sum(wineDataPCA$sdev^2)
variance

qplot(c(1:13), variance) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab("Variance") +
  ggtitle("Scree Plot of PCA") +
  ylim(0, 1) 

wineDataPCA

#Plot to visualize pair wise comparison of PCA
pairs(wineDataPCA$x)

#===================== Compute Clustering ======================================

#Compute K-means clustering on PCA
wineDataKMeans = kmeans(wineDataPCA$x[,1:2], 3)
plot(wineDataPCA$x[,1:2], col=wineDataKMeans$cluster)

expected_values <- factor(c(wineData$ï..Class))
predicted_values <- factor(c(wineDataKMeans$cluster))

confusionMatrix <- confusionMatrix(data=predicted_values, reference = expected_values)
confusionMatrix

#================== Attempt 3 - PCA with feature drop ==========================

#================ Feature engineering ==========================================

highlyCorrM <- findCorrelation(CorrelationMatrix, cutoff=0.6)
names(wineDataExcludeClass)[highlyCorrM] #Can drop "Flavanoids" and "TotalPhenols"

wineDataAfterDrop = subset(wineDataExcludeClass, select = -c(Flavanoids, TotalPhenols))
summary(wineDataAfterDrop)
plot(wineDataAfterDrop)

wineDataPCA = prcomp(wineDataAfterDrop, center = TRUE, scale = TRUE)
summary(wineDataPCA)

variance = wineDataPCA$sdev^2 / sum(wineDataPCA$sdev^2)
variance

qplot(c(1:11), variance) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab("Variance") +
  ggtitle("Scree Plot of PCA") +
  ylim(0, 1) 

wineDataPCA

#Plot to visualize pair wise comparison of PCA
pairs(wineDataPCA$x)

#===================== Compute Clustering ======================================

#Compute K-means clustering on PCA
wineDataKMeans = kmeans(wineDataPCA$x[,1:2], 3)
plot(wineDataPCA$x[,1:2], col=wineDataKMeans$cluster)

#================== Cluster evaluation =========================================

expected_values <- factor(c(wineData$ï..Class))
predicted_values <- factor(c(wineDataKMeans$cluster))

confusionMatrix <- confusionMatrix(data=predicted_values, reference = expected_values)
confusionMatrix

#====================== Attempt 4 - Selected features ==========================

#================ Feature engineering ==========================================

wineDataAfterDrop = subset(wineDataExcludeClass, select = -c(Alcohol, Ash, Flavanoids, TotalPhenols, NonFlavanoidPhenols, Hue, OD280.OD315))
summary(wineDataAfterDrop)
plot(wineDataAfterDrop)

#================= Clustering ==================================================

wineDataKMeans = kmeans(wineDataAfterDrop, 3)
wineDataKMeans

plot(wineDataAfterDrop, col=wineDataKMeans$cluster)

#================== Cluster evaluation =========================================

expected_values <- factor(c(wineData$ï..Class))
predicted_values <- factor(c(wineDataKMeans$cluster))

confusionMatrix <- confusionMatrix(data=predicted_values, reference = expected_values)
confusionMatrix

#=============================== Attempt 5 - Feature ranking ===================

#==================== Feature engineering ======================================

# prepare training scheme
control <- trainControl(method="cv", number=10)
# train the model
X <- subset(wineData, select=-ï..Class)
Y <- wineData$ï..Class
Y <- as.factor(Y)
model <- train(X, Y, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

wineDataAfterDrop = subset(wineDataExcludeClass, select = c(Flavanoids, Hue, OD280.OD315, Proline))
summary(wineDataAfterDrop)
plot(wineDataAfterDrop)

#================= Clustering ==================================================

wineDataKMeans = kmeans(wineDataAfterDrop, 3)
wineDataKMeans

plot(wineDataAfterDrop, col=wineDataKMeans$cluster)

#================== Cluster evaluation =========================================

expected_values <- factor(c(wineData$ï..Class))
predicted_values <- factor(c(wineDataKMeans$cluster))

confusionMatrix <- confusionMatrix(data=predicted_values, reference = expected_values)
confusionMatrix

correlationActualVsPredicted = cor.test(as.numeric(expected_values), as.numeric(predicted_values), method="pearson")
correlationActualVsPredicted