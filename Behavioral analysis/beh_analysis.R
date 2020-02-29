rm(list = ls()) #clear environment

library(tidyverse)
library(R.matlab)
library(lme4)
library(lmerTest)
library(reshape2)
library(data.table)
#library(spectral)
library(sjPlot)
library(kableExtra)
library(SDMTools)
library(ROCR)

#########################################################################################
#                              Data Tidying & Cleaning                                  #
#########################################################################################

#set wd to filesource, define dir
dir <- setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dir <- file.path(dir,"behavior.mat")

#read in data
B_orig <- readMat(dir)
B_orig <- data.frame(matrix(unlist(B_orig), ncol = 5, byrow=FALSE))
B <- B_orig
colnames(B, do.NULL = FALSE)
colnames(B) <- c("ID", "SDT", "Acc", "PI", "RT")





####################
# outlier analysis #
####################

## arbitrary cutoffs
#B <- filter(B, RT > 50) #alternatively remove all trials with RTs shorter than 50ms
#B <- filter(B, RT < 1200) #alternatively remove all trials with RTs shorter than 50ms


## 3-sigma from mean logRT
# first save person-specific cutoffs
#cutoffs <- B %>% group_by(ID) %>% mutate(logRT = log(RT)) %>%
#  summarize(meanlogRT = mean(logRT), sdlogRT = sd(logRT), undercut = exp(meanlogRT-3*sdlogRT), uppercut = exp(meanlogRT+3*sdlogRT))

#B <- B %>% group_by(ID) %>% mutate(logRT = log(RT)) %>%
#mutate(meanlogRT = mean(logRT), sdlogRT = sd(logRT), undercut = meanlogRT-3*sdlogRT, uppercut = meanlogRT+3*sdlogRT) %>%
#mutate(outlier = logRT < undercut | logRT > uppercut) %>% group_by(ID, SDT, Acc, PI, RT, outlier) %>%
#filter(outlier == F) %>% group_by(ID, SDT, Acc, PI, RT) %>% summarize

## Lowest and highest percentile
B <- B %>% group_by(ID) %>% mutate(logRT = log(RT)) %>%
  mutate(meanlogRT = mean(logRT), sdlogRT = sd(logRT), undercut = quantile(logRT, 0.01), uppercut = quantile(logRT, 0.99)) %>%
  mutate(outlier = logRT < undercut | logRT > uppercut) %>% group_by(ID, SDT, Acc, PI, RT, outlier) %>%
  filter(outlier == F) %>% group_by(ID, SDT, Acc, PI, RT) %>% summarize

#re-scaling is necessary for model convergence
#mean-centering is not necessary but useful for interpretation
B$RT <- scale(B$RT, center = TRUE, scale = TRUE) # rescale data by 1 SD + mean-center
#B$RT <- (B$RT - ave(B$RT, B$ID)/ave(B$RT, B$ID, FUN=sd)) # alternative: rescale data by 1 SD and mean-center subject-wise  


# duplicate df to analyze previous trials
B_post <- bind_rows(B, c(ID = NaN, SDT = NaN, Acc= NaN, PI = NaN, RT= NaN)) #ignore the warning 
B_prev <- rbind(c(ID = NaN, SDT = NaN, Acc= NaN, PI = NaN, RT= NaN), B)
colnames(B_prev) <- c("ID_prev", "SDT_prev", "Acc_prev", "PI_prev", "RT_prev")
B <- cbind(B_prev, B_post)

B <- data.frame(matrix(unlist(B), ncol = 10, byrow=FALSE)) #cbind weirdly makes a matrix out of our df
colnames(B) <- c("ID_prev", "SDT_prev", "Acc_prev", "PI_prev", "RT_prev", "ID", "SDT", "Acc", "PI", "RT")

q <- B$ID_prev != B$ID #exclude rows of "overlapping" subjects
B <- filter(B, q == F) 
#B$ID <- as.factor(B$ID) #this messes up some data manipulation
#B$ID_prev <- as.factor(B$ID_prev)



#########################################################################################
#                                 Error & SAT Analysis                                  #
#########################################################################################

B$ID <- as.factor(B$ID)
B$True_intensity <- ifelse (B$SDT == 1 |B$SDT == 2, "High", "Low")
B$Perceived_intensity <- ifelse (B$SDT == 1 |B$SDT == 3, "High", "Low")

#error analysis
EA <- B %>% group_by(ID) %>% # separate for each participant
  summarise(Mean_Acc = mean(Acc), seAcc = sd(Acc)/sqrt(n())) # calculate mean accuracy

ggplot(EA, aes(x=ID, y = Mean_Acc)) + geom_bar(stat = 'identity', color="black", fill="darkgrey") + 
  geom_errorbar(aes(ymin = Mean_Acc-seAcc, ymax = Mean_Acc+seAcc), width=.2,
                position=position_dodge(0.9))+
  xlab('Participant') + ylab('Mean Accuracy')+
  labs(title = "Mean Accuracy for each participant")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5, size = 16))
ggsave("Error analysis.png", device = "png",  scale = 1.6, dpi = 320)

# again depending on true intensity
Palette <- c("cornflowerblue", "lightsteelblue")

EA_c <- B %>% group_by(ID, True_intensity) %>% # separate for each participant
  summarise(Mean_Acc = mean(Acc), seAcc = sd(Acc)/sqrt(n())) # calculate mean accuracy
ggplot(EA_c, aes(y=Mean_Acc, x = ID, fill=True_intensity)) +
  geom_bar(stat="identity", width = 0.9, color="black", position=position_dodge())+
  geom_errorbar(aes(ymin = Mean_Acc-seAcc, ymax = Mean_Acc+seAcc), width=.2,
                position=position_dodge(0.9))+
  scale_fill_manual(values=Palette)+
  xlab("Participant")+
  ylab("Percentage correct")+
  labs(fill = "True intensity")+
  labs(title = "Mean Accuracy per true intensity for each participant")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5, size = 16))
ggsave("Error analysis conditionwise.png", device = "png",  scale = 1.6, dpi = 320)


#speed accuracy trade-off
SAT <- B %>% group_by(ID, True_intensity, Acc) %>% 
  summarise(meanRT = mean(RT), seRT = sd(RT)/sqrt(n())) 

SAT_wide <- B %>% group_by(ID, True_intensity) %>% summarise(mRT = mean(RT))
SAT_wide <- dcast(setDT(SAT_wide), ID ~ True_intensity, value.var=c("mRT"))
t.test(SAT_wide$High, SAT_wide$Low, paired = TRUE, alternative = "two.sided")

SAT_plot <- B %>% group_by(ID, True_intensity, Acc) %>% 
  summarise(meanRT = mean(RT), seRT = sd(RT)/sqrt(n())) %>%
  group_by(True_intensity, Acc) %>% 
  summarise(mRT = mean(meanRT), seRT = sd(meanRT)/sqrt(n()))
SAT_plot$Acc <- as.factor(SAT_plot$Acc)


Palette <- c("cornflowerblue", "lightsteelblue")
ggplot(SAT_plot, aes(y=mRT, x = Acc, fill=True_intensity)) +
  geom_bar(stat="identity", width = 0.9, color="black", position=position_dodge())+
  geom_errorbar(aes(ymin = mRT-seRT, ymax = mRT+seRT), width=.2,
                position=position_dodge(0.9))+
  scale_fill_manual(values=Palette)+
  coord_cartesian(ylim=c(500, 800))+
  xlab("Accuracy")+
  ylab("RT in ms")+
  labs(fill = "True intensity")+
  labs(title = "Mean RT for each condition")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5, size = 16))
ggsave("SAT.png", device = "png",  scale = 1.0, dpi = 320)



# look for main effect of error on RT
library(afex)
AOV_sat <- aov_ez("ID", "meanRT", SAT,
                  within = c("Acc", "True_intensity"))
print(AOV_sat)



#########################################################################################
#                                  FFT of Errors (WIP)                                  #
#########################################################################################

plot.frequency.spectrum <- function(X.k, xlimits=c(0,length(X.k))) {
  plot.data  <- cbind(0:(length(X.k)-1), Mod(X.k))
  
  # TODO: why this scaling is necessary?
  plot.data[2:length(X.k),2] <- 2*plot.data[2:length(X.k),2] 
  
  plot(plot.data, t="h", lwd=2, main="", 
       xlab="Frequency (Hz)", ylab="Strength", 
       xlim=xlimits, ylim=c(0,max(Mod(plot.data[,2]))))
}


z <-  B %>% group_by (ID, Acc) %>% filter (ID == 1)
z <-  z[['RT']]
class(z) 
x <- analyticFunction(z)
plot.frequency.spectrum(x)

#########################################################################################
#                                    LME Modelling I                                    #
#########################################################################################
B$TI <- as.integer(B$SDT == 1 |B$SDT == 2)


# now to MLE Logit-Regression (RI RS Model with trials as L1 vars and subjects as L2 vars) 
m1 <- lmer(RT ~ 1 + TI + PI + (1+TI  + PI | ID), data = B)
summary(m1)
#the p-value is generated from a Wald test (see documentation) - "assumptions both about
#the shape of the log-likelihood surface and about the accuracy of a chi-squared approximation to
#differences in log-likelihoods" -> uncertain validity of the assumptions, as the distribution of
#Wald-t-values needs not to be a good approximation of the z-distribution

#LRT method utilizing chi� (comparable, as glmer uses ML instead of REML like lmer) 
m1_null <- glmer(PI ~ 1  + (1 + RT_prev | ID), family = binomial, data = B)
lrt1 <- anova(m1, m1_null, simulate.p.value = TRUE)
lrt1 # the more complex has a better fit, indicating that the previous p-value may be trustworthy


#same procedure for Acc as the dv
m2 <- lmer(Acc ~ 1 + TI + (1+TI | ID), data = B)
summary(m2)
# here the fixed effect on the intercept is sign. - this cannot be tested with an LRT, as an intercept must be taken into the model. 

#LRT method utilizing chi�  
m2_null <- glmer(Acc ~ 1 + (1 + RT_prev | ID), family = binomial, data = B)
summary(m2_null)
lrt2 <- anova(m2, m2_null, simulate.p.value = T)
lrt2 # support that the fixed effect on the slope is n.s.

#nice summary table
result <- tab_model(m1, m2, show.stat = T)
result



#########################################################################################
#                                    LME Modelling II                                   #
#########################################################################################


# Optionally, let's see how high the ICC is (indicating dependent residuals) by using a RIM 
m0 <- glmer(PI ~ 1 + (1 | ID), family = binomial, data = B)
summary(m0)
tab_model(m0)
# ICC element of ]0, 1[ -> partial pooling indicated


# now to MLE Logit-Regression (RI RS Model with trials as L1 vars and subjects as L2 vars) 
m1 <- glmer(PI ~ 1 + RT_prev + (1 + RT_prev | ID), family = binomial, data = B)
summary(m1)
#the p-value is generated from a Wald test (see documentation) - "assumptions both about
#the shape of the log-likelihood surface and about the accuracy of a chi-squared approximation to
#differences in log-likelihoods" -> uncertain validity of the assumptions, as the distribution of
#Wald-t-values needs not to be a good approximation of the z-distribution

#LRT method utilizing chi� (comparable, as glmer uses ML instead of REML like lmer) 
m1_null <- glmer(PI ~ 1  + (1 + RT_prev | ID), family = binomial, data = B)
lrt1 <- anova(m1, m1_null, simulate.p.value = TRUE)
lrt1 # the more complex has a better fit, indicating that the previous p-value may be trustworthy


#same procedure for Acc as the dv
m2 <- glmer(Acc ~ 1 + RT_prev + (1 + RT_prev | ID), family = binomial, data = B)
summary(m2)
# here the fixed effect on the intercept is sign. - this cannot be tested with an LRT, as an intercept must be taken into the model. 

#LRT method utilizing chi�  
m2_null <- glmer(Acc ~ 1 + (1 + RT_prev | ID), family = binomial, data = B)
summary(m2_null)
lrt2 <- anova(m2, m2_null, simulate.p.value = T)
lrt2 # support that the fixed effect on the slope is n.s.

#nice summary table
result <- tab_model(m1, m2, show.stat = T)
result

# plot ROC
confusion.matrix(B$Acc, fitted(m1))
confusion.matrix(B$Acc, fitted(m2))


pred <- prediction(fitted(m1), B$PI)
perf1 <- performance(pred, "tpr", "fpr")
pred <- prediction(fitted(m2), B$Acc)
perf2 <- performance(pred, "tpr", "fpr")


plot(perf1, col = "blue", lwd = 2) 
par(new=TRUE)
plot(perf2, col = "red", lwd = 2)
abline(0,1, lty = 2, col = "gray") 

#============================#
#      Additonal plots       #
#============================#
B_plot <- B_orig
colnames(B_plot, do.NULL = FALSE)
colnames(B_plot) <- c("ID", "SDT", "Acc", "PI", "RT")
B_plot <-B_plot[order(B_plot$RT),]
B_plot <-B_plot[order(B_plot$ID),]
lv1 <- 1:17
lv1 <- lv1[-13]
lv2 <- 18:33

# loop plot
par(mfrow = c(4, 4))
for (i in lv2) {
   p1 <- filter(B_plot, ID == i)
  # png(paste(i, "_RT_sorted.png", sep="")) 
   plot(p1$RT, main=i,
      ylab="RT",xlim=c(0, 1000),  ## with c()
      ylim=c(0, 1500),  ## with c()
      )
  #dev.off()
  }

