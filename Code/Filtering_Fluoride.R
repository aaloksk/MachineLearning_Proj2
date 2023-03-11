#Written by Aalok Sharma Kafle
#Filtering the wells in TEDB groundwater data

#Import necessary libraries


#Setting working directory to the folder with original/extracted GWDB database
path <- 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Proj1\\WD\\GWDBDownload\\GWDBDownload'
setwd(path)

#Import master database of water quality in major aquifers
a <- read.csv('WaterQualityMinor.txt', sep = "|", quote = "", row.names = NULL, stringsAsFactors = FALSE)
unique(a$Aquifer)

#Filtering for Dockum Aquifer Data
b1 <- a[a$Aquifer == "Dockum",]
b2 <- a[a$Aquifer == "Edwards-Trinity (High Plains)",]
b <- rbind(b1,b2)

b <- b[b$SampleYear>= 1990,]


#Looking at frequency measurement
zz <- data.frame(table(b$ParameterDescription))


#data_Ar
#d1<- b[b$ParameterDescription == 'ARSENIC, DISSOLVED (UG/L AS AS)',]


#data_Fl 
d2<- b[b$ParameterDescription == 'FLUORIDE, DISSOLVED (MG/L AS F)',]



#res <- data.frame(tapply(d1$ParameterValue, d1$StateWellNumber, mean))
#res_sd <- data.frame(tapply(d1$ParameterValue, d1$StateWellNumber, sd))
#res_max <- data.frame(tapply(d1$ParameterValue, d1$StateWellNumber, max))
#Ar <- data.frame(rownames(res), res$tapply.d1.ParameterValue..d1.StateWellNumber..mean., res_sd$tapply.d1.ParameterValue..d1.StateWellNumber..sd.,res_max$tapply.d1.ParameterValue..d1.StateWellNumber..max.)
##colnames(Ar) <- c("WellID", "Avg_Ar", "Ar_SD","Ar_Max")
#Ar$rat <- Ar$Ar_Max/Ar$Avg_Ar
#Ar2 <- Ar[Ar$rat <= 1.75,]




res2 <- data.frame(tapply(d2$ParameterValue, d2$StateWellNumber, mean))
res2$SD <- tapply(d2$ParameterValue, d2$StateWellNumber, sd)
res2$MAX <- tapply(d2$ParameterValue, d2$StateWellNumber, max)
Fl <- data.frame(rownames(res2), res2[,1], res2[,2], res2[,3])
colnames(Fl) <- c("WellID", "Fl_Avg", "Fl_SD", "Fl_Max")
Fl$check <- Fl$Fl_Max/Fl$Fl_Avg






#List of unique wells
wells <- unique(Fl$WellID)



#Import the Well Main File
wm <- read.csv('WellMain.txt', sep = "|", quote = "", row.names = NULL, stringsAsFactors = FALSE)

#Getting only pre-defined unique wells with at least one nitrate measurement since 1980
wm2 <- wm[wm$StateWellNumber %in% wells,]


#Dataframe with all necessary elements
df <- data.frame(Fl$WellID, wm2$LatitudeDD, wm2$LongitudeDD, wm2$WellDepth, Fl$Fl_Avg)
colnames(df) <- c("WellID", "LatDD", "LonDD", "Well_Depth","Avg_Fl")


#Removing rows with no data of well depth
df_all <- df[!is.na(df$Well_Depth),]



path2 <- 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Project2'
setwd(path2)
write.csv(df_all, "Wells.csv", row.names = FALSE)




cor.test(df_all$Avg_Fl, df_all$Well_Depth, method = "pearson")






df_all <- data.frame(wm2$StateWellNumber, wm2$LatitudeDD, wm2$LongitudeDD,df$No_of_Measurement ,df$Avg_Hardness, wm2$WellDepth)
colnames(df_all) <- c("StateWellNumber","LatDD","LongDD","Meas_Count","Avg_Hardness","Well Depth")
head(df_all)

#Removing rows with no data of well depth
df_all <- df_all[!is.na(df_all$`Well Depth`),]

#Removing rows with 0 value of Average Nitrate
df_all <- df_all[df_all$Avg_Hardness !=0,]


#Plotting Nitrate_Nitrogen with Well-Depth
y <- df_all$Avg_Hardness #Y variable
ylab <- c("Hardness") #Y-axis label
x <- df_all$`Well Depth` #X-variable
xlab <- c("Well Depth") #X-axis label
plot(x,y , xlab = xlab, ylab = ylab) #Actual Plot
grid()


#Data distribution
nrow(df2[df2$Avg_Hardness>300,]) / nrow(df2)









#path2 <- 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Project2'
#setwd(path2)
#write.csv(df_all, "Wells.csv", row.names = FALSE)

#Extracting Water Levels Major File from GWDB#
#w_lev <- read.csv('WaterLevelsMinor.txt', sep = "|", quote = "", row.names = NULL, stringsAsFactors = FALSE)
#Getting all the remaining unique wells
#wells2 <- unique(df_all$StateWellNumber)
#Getting only pre-defined unique wells
#w_lev2 <- w_lev[w_lev$StateWellNumber %in% wells2,]
#Filtering for the data that are measured post 1980.
#w_lev3 <- w_lev2[w_lev2$MeasurementYear >1990,]
#wells3 <- unique(w_lev3$StateWellNumber)






