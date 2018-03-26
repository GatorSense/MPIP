# Statistical analysis script
# 
# Authors:
#   Anand Seethepalli
#   Computer Vision Specialist
#   Root Phenomics,
#   Noble Research Institute, Ardmore, Oklahoma
#   Email: aseethepalli@noble.org
#          anand_seethepalli@yahoo.co.in
# 
#   Dr. Larry York
#   Assistant Professor,
#   Root Phenomics,
#   Noble Research Institute, Ardmore, Oklahoma
#   Email: lmyork@noble.org


library(stats)
library(purrr)
library(dplyr)
library(Matrix)
library(lme4)
library(tibble)
library(reshape2)
library(ggplot2)
library(tidyr)
library(grid)
library(gridExtra)

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
    library(grid)
    plots <- c(list(...), plotlist)
    numPlots = length(plots)
    if (is.null(layout)) {
        layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                         ncol = cols, nrow = ceiling(numPlots/cols))
    }
    if (numPlots==1) {
        print(plots[[1]])
    } else {
        grid.newpage()
        pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
        for (i in 1:numPlots) {
            matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
            print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                            layout.pos.col = matchidx$col))
        }
    }
}

# Function to get genotype information.
# This function may be changed for each experiment.
get_factor_info <- function(dataframe)
{
    nsamples <- length(dataframe[[1]]);
    genotypes <- rep(0, nsamples);
    blocks <- rep(0, nsamples);
    cams <- rep(0, nsamples);
    plantnos <- rep(0, nsamples);
    
    for (idx in 1 : nsamples)
    {
        plotval <- as.integer(substr(dataframe[[1]][idx], 3, 6));
        blockval <- floor(plotval / 1000);
        genotypeval <- plotval - (blockval - 1) * 1000;
        camval <- as.integer(substr(dataframe[[1]][idx], 10, 10));
        plantno <- as.integer(substr(dataframe[[1]][idx], 8, 8));
        
        genotypes[idx] <- genotypeval;
        blocks[idx] <- blockval;
        cams[idx] <- camval;
        plantnos[idx] <- plantno;
    }
    
    return(data.frame(cbind(genotypes, blocks, cams, plantnos)));
}

# Function to get plot averages of features extracted.
# Optionally exclude invalid images from camera 5.
get_plot_averaged_data <- function(dataframe, factorframe, includeinvalid = TRUE)
{
    factordata <- data.frame();
    result <- data.frame();
    
    if (includeinvalid == TRUE)
    {
        factordata <- cbind(factors[1:length(colnames(factors)) - 1], maindata[3:length(colnames(maindata))]);
        result <- group_by(factordata, genotypes, blocks, cams) %>% summarise_all(funs(mean(., na.rm = TRUE)));
    }
    else
    {
        factordata <- cbind(factors[1:length(colnames(factors)) - 1], maindata[2:length(colnames(maindata))]);
        factordata <- filter(factordata, Valid == 1);
        factordata <- factordata[-4];
        result <- group_by(factordata, genotypes, blocks, cams) %>% summarise_all(funs(mean(., na.rm = TRUE)));
    }
    
    return(result);
}

############################ Statistical functions ########################
# Function to get heritabilities for every extracted feature and append
# to the results data frame for plotting later.
compute_heritability <- function(dataframe, resultframe, camera = 0, manovaanalysis = 0)
{
    filtered_data <- data.frame();
    hframe <-data.frame();
    cams <- c("Left 1", "Right 1", "Left 2", "Center", "Right 2");
    camstr <- "";
    
    if (manovaanalysis == 0)
    {
        if (camera != 0)
        {
            filtered_data <- filter(dataframe, Camera == camera);
            camstr <- cams[camera];
            filtered_data <- filtered_data[-3];
        } else
        {
            filtered_data <- dataframe[-3];
            filtered_data <- group_by(filtered_data, Genotype, Block) %>% summarise_all(funs(mean(., na.rm = TRUE)));
            
            camstr <- "Combined";
        }
        
        filtered_data[[1]] <- factor(filtered_data[[1]]);
        filtered_data[[2]] <- factor(filtered_data[[2]]);
        
        result <- rep(0, length(colnames(filtered_data)) - 2);
        
        for (i in 3:length(colnames(filtered_data)))
        {
            summarytable <- summary(lmer(filtered_data[[i]] ~ Block + (1|Genotype), data = filtered_data, na.action=na.omit));
            varcor <- as.data.frame(summarytable$varcor);
            
            #the 4 needs updating for number of replicates
            result[i - 2] <- varcor$vcov[1]/(varcor$vcov[1]+varcor$vcov[2] / length(unique(dataframe$Block)));
        }
        
        hframe <- cbind(resultframe, result, stringsAsFactors = FALSE);
        colnames(hframe)[length(colnames(hframe))] <- camstr; #paste("Heritability", camstr, sep="-");
    } else
    {
        result <- map(dataframe[4:ncol(dataframe)], function(x) { 
                t <- cbind(dataframe[1:3], a=x) %>% spread(., Camera, a) %>% as.data.frame();
                colnames(t)[3:ncol(t)] <- paste("a", colnames(t)[3:ncol(t)], sep="");
                
                s <- summary(manova(cbind(t$a1, t$a2, t$a3, t$a4, t$a5) ~ Block + Genotype, data = t), test="Pillai");
                #return(1-s[[4]][17]); # For p-value
                return (s[[4]][2] * s[[4]][8] / (s[[4]][2] * s[[4]][8] + s[[4]][3])); # For effect-size
                #return((5-s[[4]][5])/5); # Formula for heritability
            }) %>% as.numeric();
        hframe <- cbind(resultframe, result, stringsAsFactors = FALSE);
        colnames(hframe)[length(colnames(hframe))] <- "Combined - Manova";
    }
    
    return(hframe) 
}

compute_meanvars <- function(dataframe, resultframe, camera = 0, manual = 0, comb = 0)
{
    filtered_data <- data.frame();
    hframe <-data.frame();
    cams <- c("L1", "R1", "L2", "C", "R2");
    camstr <- "";
    
    if (camera != 0)
    {
        filtered_data <- filter(dataframe, Camera == camera);
        camstr <- cams[camera];
        filtered_data <- filtered_data[-3];
    } else
    {
        if (manual != 0)
        {
            filtered_data <- dataframe;
            #filtered_data <- group_by(filtered_data, Genotype, Block) %>% summarise_all(funs(mean(., na.rm = TRUE)));
            camstr <- "M";
        } else
        {
            if (comb != 0)
            {
                filtered_data <- dataframe;
                #filtered_data <- group_by(filtered_data, Genotype, Block) %>% summarise_all(funs(mean(., na.rm = TRUE)));
                camstr <- "CN";
            } else
            {
                filtered_data <- dataframe[-3];
                filtered_data <- group_by(filtered_data, Genotype, Block) %>% summarise_all(funs(mean(., na.rm = TRUE)));
                
                camstr <- "AV";
            }
        }
    }
    
    filtered_data[[1]] <- factor(filtered_data[[1]]);
    filtered_data[[2]] <- factor(filtered_data[[2]]);
    
    result <- map(filtered_data[3:ncol(filtered_data)], function(x2) { 
        return (c(mean(x2, na.rm = TRUE), sd(x2, na.rm = TRUE)));
    }) %>% as.data.frame(check.names=FALSE, stringsAsFactors = FALSE) %>% t();
    
    #result <- data.frame(lapply(result[1:length(result[[1]]),1:2], as.numeric), check.names=FALSE);
    
    result <- data.frame(cbind(Features = colnames(filtered_data)[3:ncol(filtered_data)], Camera = rep(camstr, nrow(result)), result), stringsAsFactors = FALSE);
    colnames(result)[3] <- "Mean";
    colnames(result)[4] <- "SD";
    
    hframe <- rbind(resultframe, result, stringsAsFactors = FALSE);
    
    return(hframe) 
}

compute_meanvars_table <- function(dataframe, resultframe, camera = 0, manual = 0, comb = 0)
{
    filtered_data <- data.frame();
    hframe <-data.frame();
    cams <- c("L1", "R1", "L2", "C", "R2");
    camstr <- "";
    
    if (camera != 0)
    {
        filtered_data <- filter(dataframe, Camera == camera);
        camstr <- cams[camera];
        filtered_data <- filtered_data[-3];
    } else
    {
        if (manual != 0)
        {
            filtered_data <- dataframe;
            #filtered_data <- group_by(filtered_data, Genotype, Block) %>% summarise_all(funs(mean(., na.rm = TRUE)));
            camstr <- "M";
        } else
        {
            if (comb != 0)
            {
                filtered_data <- dataframe;
                #filtered_data <- group_by(filtered_data, Genotype, Block) %>% summarise_all(funs(mean(., na.rm = TRUE)));
                camstr <- "CN";
            } else
            {
                filtered_data <- dataframe[-3];
                filtered_data <- group_by(filtered_data, Genotype, Block) %>% summarise_all(funs(mean(., na.rm = TRUE)));
                
                camstr <- "AV";
            }
        }
    }
    
    filtered_data[[1]] <- factor(filtered_data[[1]]);
    filtered_data[[2]] <- factor(filtered_data[[2]]);
    
    result <- map(filtered_data[3:ncol(filtered_data)], function(x2) { 
        return (c(round(mean(x2, na.rm = TRUE), 2), round(sd(x2, na.rm = TRUE), 2), 
                  round(min(x2, na.rm = TRUE), 2), round(max(x2, na.rm = TRUE), 2)));
    }) %>% as.data.frame(check.names=FALSE, stringsAsFactors = FALSE) %>% t();
    
    #result <- data.frame(lapply(result[1:length(result[[1]]),1:2], as.numeric), check.names=FALSE);
    
    result <- data.frame(cbind(Features = colnames(filtered_data)[3:ncol(filtered_data)], Camera = rep(camstr, nrow(result)), result), stringsAsFactors = FALSE);
    colnames(result)[3] <- "Mean";
    colnames(result)[4] <- "SD";
    colnames(result)[5] <- "Min";
    colnames(result)[6] <- "Max";
    
    hframe <- rbind(resultframe, result, stringsAsFactors = FALSE);
    
    return(hframe) 
}

compute_cor <- function(x, y, camera=0, getavgfeatures = 1, avgcor = 1, maxvals = 1)
{
    filtered_data <- data.frame();
    hframe <-data.frame();

    if (camera != 0)
    {
        filtered_data <- filter(x, Camera == camera);
        filtered_data <- filtered_data[-5] %>% arrange(., `Plot Number`, Block, `Plant Number`, Genotype);
        y2 <- arrange(y, `Plot Number`, Block, `Plant Number`, Genotype);
        hframe <- cor(filtered_data[5:ncol(filtered_data)], y2[5:ncol(y2)], use = "pairwise.complete.obs") %>% as.data.frame(., check.names=FALSE);
    } else
    {
        if (getavgfeatures == 1)
        {
            filtered_data <- x[-5];
            filtered_data <- group_by(filtered_data, `Plot Number`, Block, `Plant Number`, Genotype) %>% summarise_all(funs(mean(., na.rm = TRUE))) %>% arrange(., `Plot Number`, Block, `Plant Number`, Genotype);
            y2 <- arrange(y, `Plot Number`, Block, `Plant Number`, Genotype);
            hframe <- cor(filtered_data[5:ncol(filtered_data)], y2[5:ncol(y2)], use = "pairwise.complete.obs") %>% as.data.frame(., check.names=FALSE);
        } else
        {
            if (avgcor == 1)
            {
                result <- map(x[6:ncol(x)], function(x2) { 
                    t <- cbind(x[1:5], a=x2) %>% 
                         spread(., Camera, a) %>%
                         arrange(., `Plot Number`, Block, `Plant Number`, Genotype);
                    t2 <- arrange(y, `Plot Number`, Block, `Plant Number`, Genotype);
                    co <- cor(t[5:ncol(t)], t2[5:ncol(t2)], use = "pairwise.complete.obs");
                    co <- colMeans(co);
                    co <- as.data.frame(co, check.names=FALSE);
                    return(co);
                });
                
                hframe <- as.data.frame(t(as.data.frame(result, check.names=FALSE)));
                rownames(hframe) <- colnames(x)[6:ncol(x)];
            } else
            {
                if (maxvals == 1)
                {
                    result <- map(x[6:ncol(x)], function(x2) { 
                        t <- cbind(x[1:5], a=x2) %>% 
                            spread(., Camera, a) %>%
                            arrange(., `Plot Number`, Block, `Plant Number`, Genotype);
                        t2 <- arrange(y, `Plot Number`, Block, `Plant Number`, Genotype);
                        co <- cor(t[5:ncol(t)], t2[5:ncol(t2)], use = "pairwise.complete.obs") %>% 
                            as.data.frame(., check.names=FALSE) %>% summarise_all(., funs(max)) %>% t();
                        return(co);
                    });
                    
                    #hframe <- t(as.data.frame(result, check.names=FALSE));
                    hframe <- as.data.frame(t(as.data.frame(result, check.names=FALSE)));
                    rownames(hframe) <- colnames(x)[6:ncol(x)];
                } else
                {
                    result <- map(x[6:ncol(x)], function(x2) { 
                        t <- cbind(x[1:5], a=x2) %>% 
                            spread(., Camera, a) %>%
                            arrange(., `Plot Number`, Block, `Plant Number`, Genotype);
                        t2 <- arrange(y, `Plot Number`, Block, `Plant Number`, Genotype);
                        co <- cor(t[5:ncol(t)], t2[5:ncol(t2)], use = "pairwise.complete.obs") %>% 
                            as.data.frame(., check.names=FALSE) %>% summarise_all(., funs(which.max)) %>% t();
                        return(co);
                    });
                    
                    #hframe <- t(as.data.frame(result, check.names=FALSE));
                    hframe <- as.data.frame(t(as.data.frame(result, check.names=FALSE)));
                    rownames(hframe) <- colnames(x)[6:ncol(x)];
                }
            }
        }
    }
    
    return(hframe);
}

compute_corpval <- function(x, y, camera=0, getavgfeatures = 1, avgcor = 1)
{
    cor.test.p <- function(x3, y3){
        FUN <- function(x2, y2) cor.test(x2, y2, use = "pairwise.complete.obs")[["p.value"]]
        z <- outer(
            colnames(x3), 
            colnames(y3), 
            Vectorize(function(i,j) FUN(x3[[i]], y3[[j]]))
        )
        dimnames(z) <- list(colnames(x3), colnames(y3))
        return(z);
    }
    
    filtered_data <- data.frame();
    hframe <-data.frame();
    
    if (camera != 0)
    {
        filtered_data <- filter(x, Camera == camera);
        filtered_data <- filtered_data[-5] %>% arrange(., `Plot Number`, Block, `Plant Number`, Genotype);
        y2 <- arrange(y, `Plot Number`, Block, `Plant Number`, Genotype);
        hframe <- cor.test.p(filtered_data[5:ncol(filtered_data)], y2[5:ncol(y2)]) %>% as.data.frame(., check.names=FALSE);
    } else
    {
        if (getavgfeatures == 1)
        {
            filtered_data <- x[-5];
            filtered_data <- group_by(filtered_data, `Plot Number`, Block, `Plant Number`, Genotype) %>% 
                summarise_all(funs(mean(., na.rm = TRUE))) %>% 
                arrange(., `Plot Number`, Block, `Plant Number`, Genotype);
            y2 <- arrange(y, `Plot Number`, Block, `Plant Number`, Genotype);
            hframe <- cor.test.p(filtered_data[5:ncol(filtered_data)], y2[5:ncol(y2)]) %>% as.data.frame(., check.names=FALSE);
        } else
        {
            if (avgcor == 1)
            {
                result <- map(x[6:ncol(x)], function(x2) { 
                    t <- cbind(x[1:5], a=x2) %>% 
                        spread(., Camera, a) %>%
                        arrange(., `Plot Number`, Block, `Plant Number`, Genotype);
                    t2 <- arrange(y, `Plot Number`, Block, `Plant Number`, Genotype);
                    co <- cor.test.p(t[5:ncol(t)], t2[5:ncol(t2)]);
                    co <- colMeans(co);
                    co <- as.data.frame(co, check.names=FALSE);
                    return(co);
                });
                
                hframe <- as.data.frame(t(as.data.frame(result, check.names=FALSE)));
                rownames(hframe) <- colnames(x)[6:ncol(x)];
            } else
            {
                result <- map(x[6:ncol(x)], function(x2) { 
                    t <- cbind(x[1:5], a=x2) %>% 
                        spread(., Camera, a) %>%
                        arrange(., `Plot Number`, Block, `Plant Number`, Genotype);
                    t2 <- arrange(y, `Plot Number`, Block, `Plant Number`, Genotype);
                    co <- cor(t[5:ncol(t)], t2[5:ncol(t2)], use = "pairwise.complete.obs") %>% 
                        as.data.frame(., check.names=FALSE) %>% summarise_all(., funs(which.max)) %>% t();
                    co2 <- cor.test.p(t[5:ncol(t)], t2[5:ncol(t2)]) %>% 
                        as.data.frame(., check.names=FALSE);
                    co3 <- co2[cbind(co, c(1:nrow(co)))] %>% as.data.frame() %>% t();
                    colnames(co3) <- colnames(co2);
                    return(t(co3));
                });
                
                hframe <- as.data.frame(t(as.data.frame(result, check.names=FALSE)));
                rownames(hframe) <- colnames(x)[6:ncol(x)];
            }
        }
    }
    
    return(hframe);
}

compute_camcor <- function(x)
{
    result <- map(x[6:ncol(x)], function(x2) { 
        t <- cbind(x[1:5], a=x2) %>% 
            spread(., Camera, a) %>%
            arrange(., `Plot Number`, Block, `Plant Number`, Genotype);
        co <- cor(t[5:ncol(t)], t[5:ncol(t)], use = "pairwise.complete.obs");
        co2 <- c(mean(co[upper.tri(co)]), sd(co[upper.tri(co)])) %>% as.data.frame() %>% t();
        colnames(co2) <- c("means", "sds");
        return(t(co2));
    });
    
    hframe <- as.data.frame(t(as.data.frame(result, check.names=FALSE)));
    hframe$feats <- colnames(x)[6:ncol(x)];
    
    return(hframe);
}

compute_camcor_with_means <- function(x)
{
    result <- map(x[4:ncol(x)], function(x2) { 
        t <- cbind(x[1:3], a=x2) %>% 
            spread(., Camera, a) %>%
            arrange(., Block, Genotype);
        co <- cor(t[3:ncol(t)], t[3:ncol(t)], use = "pairwise.complete.obs");
        co2 <- c(mean(co[upper.tri(co)]), sd(co[upper.tri(co)])) %>% as.data.frame() %>% t();
        colnames(co2) <- c("means", "sds");
        return(t(co2));
    });
    
    hframe <- as.data.frame(t(as.data.frame(result, check.names=FALSE)));
    hframe$feats <- colnames(x)[4:ncol(x)];
    
    return(hframe);
}

create_heatmap <- function(dfr, titlestr, nbins = 14, lo=-0.7, hi=0.7)
{
    mat <- as.matrix(dfr, check.names=FALSE, dimnames=list(rownames(dfr), colnames(dfr)));
    mat <- mat[ nrow(mat):1, ];
    pdata <- melt(mat);
    
    p = ggplot(aes(x=Var2, y=Var1), data=pdata) +
        geom_tile(aes(fill=value)) +
        xlab("") + ylab("") + coord_fixed(ratio = 0.8) +
        ggtitle(titlestr) +
        #scale_fill_continuous(limits=c(lo, hi), guide = guide_colorbar(barwidth = 0.5, barheight = 10, nbin = 4)) + 
        scale_fill_gradient2(limits=c(lo, hi), breaks = (-(nbins/2):(nbins/2)) / ((nbins/2) * (1 / hi)), name="") + 
        theme(axis.title.x = element_text(size=15, face="bold"), 
              axis.title.y = element_text(size=15, face="bold"), 
              legend.key.size=unit(0.5,"cm"),
              axis.text.x = element_text(angle = 45, hjust = 1, size=13), 
              axis.text.y = element_text(size=13),
              text=element_text(size=12),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              plot.title = element_text(hjust = 0.5, size=14, face="bold"),
              legend.position = "right",
              legend.key.height = unit(2, "cm"));
    
    return(p);
}

create_discrete_heatmap <- function(dfr, titlestr, colorlist, labellist, levellist)
{
    mat <- as.matrix(dfr, check.names=FALSE, dimnames=list(rownames(dfr), colnames(dfr)));
    mat <- mat[ nrow(mat):1, ];
    pdata = melt(mat);
    
    #colorslist = c("darkolivegreen3", "deepskyblue", "darkolivegreen4", "firebrick", "dodgerblue3");
    pdata$value = factor(pdata$value, levels=levellist);
    p = ggplot(aes(x=Var2, y=Var1), data=pdata) +
        geom_tile(aes(fill=factor(value))) +
        xlab("") + ylab("") + coord_fixed(ratio = 0.8) + 
        ggtitle(titlestr) +
        scale_fill_manual(name="Camera", labels = labellist, values=colorlist, drop = FALSE) + 
        theme(axis.title.x = element_text(size=15, face="bold"), 
              axis.title.y = element_text(size=15, face="bold"), 
              legend.key.size=unit(0.5,"cm"),
              axis.text.x = element_text(angle = 45, hjust = 1, size=13), 
              axis.text.y = element_text(size=13),
              text=element_text(size=12),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              plot.title = element_text(hjust = 0.5, size=14, face="bold"),
              legend.position = "right");
    
    return(p);
}

compute_etasq <- function(dataframe, pnames)
{
    dtf <- dataframe;
    
    result <- map(dtf[4:ncol(dtf)], function(x) { 
        t <- cbind(dtf[1:3], a=x);
        t <- t[complete.cases(t), ];
        
        lmv = lm(t$a ~ Genotype + Block + Camera, data=dtf, na.action=na.omit)
        aovv = anova(lmv);
        denom <- (aovv[[2]][1] + aovv[[2]][2] + aovv[[2]][3] + aovv[[2]][4]);
        pvals <- c(1, aovv[[5]][3], aovv[[5]][2], aovv[[5]][1]);
        signv = ifelse(pvals < 0.05, ifelse(pvals < 0.01,ifelse(pvals < 0.001, "\n***", "\n**"),"\n*"), "");
        
        retframe <- data.frame(cbind(Factors = c("Residual", "Camera", "Block", "Genotype"), 
                                     Effectsize=c((aovv[[2]][4] / denom), 
                                                  (aovv[[2]][3] / denom), 
                                                  (aovv[[2]][2] / denom),
                                                  (aovv[[2]][1] / denom)), 
                                     Cumeffectsize=c((aovv[[2]][1] / denom) + (aovv[[2]][2] / denom) + (aovv[[2]][3] / denom),
                                                     (aovv[[2]][1] / denom) + (aovv[[2]][2] / denom), (aovv[[2]][1] / denom), 0), 
                                     Significance=signv), stringsAsFactors = FALSE);
        
        return (retframe);
    });
    
    hframe <- result %>% map_df(bind_rows);
    hframe <- data.frame(cbind(Features = rep(colnames(dataframe)[4:length(colnames(dataframe))], each=4), 
                               hframe, stringsAsFactors = FALSE));
    hframe[3:4] <- data.frame(lapply(hframe[1:length(hframe[[1]]),3:4], as.numeric), 
                              check.names=FALSE, stringsAsFactors = FALSE);
    hframe$Features <- factor(hframe$Features, levels = pnames, ordered = TRUE);
    hframe$Factors <- factor(hframe$Factors, levels = rev(c("Genotype", "Block", "Camera", "Residual")), ordered = TRUE);
    return(hframe) 
}

############################ Main script start ##############################
# Set the variable UseDIRTFeatures to 1, if DIRT features need to be analyzed.
UseDIRTFeatures = 0;
setwd("C:/Users/aseethepalli/Desktop/rscript");

if (UseDIRTFeatures == 0)
{
    maindata <- read.csv(file="features_2016_ifeatures.csv", stringsAsFactors = FALSE, check.names=FALSE, na.strings = c("NA", "na"));
    maindata <- data.frame(lapply(maindata[1:length(maindata[[1]]),1:43], as.numeric), check.names=FALSE);
    
    # Use only the Historical data of Rollins Bottoms field from Hussien's experimental data. Also to remove 
    # unnecessary features and modify existing ones.
    imgdata <- maindata[1:2400, -37:-42];
    imgdata2 <- cbind(imgdata[1:19], 
                     imgdata[[32]] + imgdata[[33]], 
                     imgdata[[31]] + imgdata[[34]], 
                     imgdata[[30]] + imgdata[[35]],
                     (imgdata[[32]] + imgdata[[33]]) / (imgdata[[30]] + imgdata[[35]]),
                     imgdata[[20]] + imgdata[[21]] + imgdata[[22]],
                     imgdata[[23]] + imgdata[[24]] + imgdata[[25]] + imgdata[[26]],
                     imgdata[[27]] + imgdata[[28]] + imgdata[[29]],
                     (imgdata[[20]] + imgdata[[21]] + imgdata[[22]]) / (imgdata[[27]] + imgdata[[28]] + imgdata[[29]]),
                     imgdata[36:37]);
    colnames(imgdata2)[20] <- "Shallow Angle Freq.";
    colnames(imgdata2)[21] <- "Medium Angle Freq.";
    colnames(imgdata2)[22] <- "Steep Angle Freq.";
    colnames(imgdata2)[23] <- "Shallowness Index";
    colnames(imgdata2)[24] <- "Fine Radius Freq.";
    colnames(imgdata2)[25] <- "Medium Radius Freq.";
    colnames(imgdata2)[26] <- "Coarse Radius Freq.";
    colnames(imgdata2)[27] <- "Fineness Index";
    
    # To convert pixel units to physical units
    imgdata3 <-imgdata2;
    lg1x = (((0.61 - 0.0352080) / 0.0352080) * (14.0 / 3264));
    ls110 = (((0.61 - 0.0135860) / 0.0135860) * (5.58 / 3000));
    ag1x = lg1x * lg1x;
    as110 = ls110 * ls110;
    vg1x = lg1x * lg1x * lg1x;
    vs110 = ls110 * ls110 * ls110;
    
    imgdata3[[7]] <- as.numeric(map_if(imgdata3[[7]], imgdata3[[4]] < 3, function(x) {return(x * lg1x)}));
    imgdata3[[7]] <- as.numeric(map_if(imgdata3[[7]], imgdata3[[4]] >= 3, function(x) {return(x * ls110)}));
    
    imgdata3[[8]] <- as.numeric(map_if(imgdata3[[8]], imgdata3[[4]] < 3, function(x) {return(x * lg1x)}));
    imgdata3[[8]] <- as.numeric(map_if(imgdata3[[8]], imgdata3[[4]] >= 3, function(x) {return(x * ls110)}));
    
    imgdata3[[9]] <- as.numeric(map_if(imgdata3[[9]], imgdata3[[4]] < 3, function(x) {return(x * lg1x)}));
    imgdata3[[9]] <- as.numeric(map_if(imgdata3[[9]], imgdata3[[4]] >= 3, function(x) {return(x * ls110)}));
    
    imgdata3[[11]] <- as.numeric(map_if(imgdata3[[11]], imgdata3[[4]] < 3, function(x) {return(x * ag1x)}));
    imgdata3[[11]] <- as.numeric(map_if(imgdata3[[11]], imgdata3[[4]] >= 3, function(x) {return(x * as110)}));
    
    imgdata3[[12]] <- as.numeric(map_if(imgdata3[[12]], imgdata3[[4]] < 3, function(x) {return(x * ag1x)}));
    imgdata3[[12]] <- as.numeric(map_if(imgdata3[[12]], imgdata3[[4]] >= 3, function(x) {return(x * as110)}));
    
    imgdata3[[14]] <- as.numeric(map_if(imgdata3[[14]], imgdata3[[4]] < 3, function(x) {return(x * lg1x)}));
    imgdata3[[14]] <- as.numeric(map_if(imgdata3[[14]], imgdata3[[4]] >= 3, function(x) {return(x * ls110)}));
    
    imgdata3[[15]] <- as.numeric(map_if(imgdata3[[15]], imgdata3[[4]] < 3, function(x) {return(x * lg1x)}));
    imgdata3[[15]] <- as.numeric(map_if(imgdata3[[15]], imgdata3[[4]] >= 3, function(x) {return(x * ls110)}));
    
    imgdata3[[16]] <- as.numeric(map_if(imgdata3[[16]], imgdata3[[4]] < 3, function(x) {return(x * vg1x)}));
    imgdata3[[16]] <- as.numeric(map_if(imgdata3[[16]], imgdata3[[4]] >= 3, function(x) {return(x * vs110)}));
    
    imgdata3[[17]] <- as.numeric(map_if(imgdata3[[17]], imgdata3[[4]] < 3, function(x) {return(x * ag1x)}));
    imgdata3[[17]] <- as.numeric(map_if(imgdata3[[17]], imgdata3[[4]] >= 3, function(x) {return(x * as110)}));
    
    imgdata3[[18]] <- as.numeric(map_if(imgdata3[[18]], imgdata3[[4]] < 3, function(x) {return(x * lg1x)}));
    imgdata3[[18]] <- as.numeric(map_if(imgdata3[[18]], imgdata3[[4]] >= 3, function(x) {return(x * ls110)}));
    
    imgdata3[[19]] <- as.numeric(map_if(imgdata3[[19]], imgdata3[[4]] < 3, function(x) {return(x * ag1x)}));
    imgdata3[[19]] <- as.numeric(map_if(imgdata3[[19]], imgdata3[[4]] >= 3, function(x) {return(x * as110)}));
    imgdata3 <- add_column(imgdata3, floor(imgdata3[[1]] / 1000), .after = 1);
    colnames(imgdata3)[2] <- "Block";
    imgdata3[[1]] <- (imgdata3[[1]] - ((imgdata3[[2]] - 1) * 1000));
    imgdata <- imgdata3;
    remove(imgdata2, imgdata3, lg1x, ls110, ag1x, as110, vg1x, vs110);
    imgdata[[1]] <- factor(imgdata[[1]]);
    imgdata[[2]] <- factor(imgdata[[2]]);
    imgdata[[4]] <- factor(imgdata[[4]]);
    imgdata[[5]] <- factor(imgdata[[5]]);
    imgdataorig <- imgdata;
    imgdata <- group_by(imgdata, `Plot Number`, Block, Genotype, Camera) %>% summarise_all(funs(mean(., na.rm = TRUE)));
    imgdata <- imgdata[-5];
    imgdata <- imgdata[-1];
    
    temp <- imgdata[-3];
    combdata <- group_by(temp, Block, Genotype) %>% 
                summarise(`Max. Max. Width` = max(`Max. width`, na.rm = TRUE),
                       `Min. Max. Width` = min(`Max. width`, na.rm = TRUE),
                       Eccentricity = max(`Max. width`, na.rm = TRUE) / min(`Max. width`, na.rm = TRUE),
                       `Max. Max. Width-to-Avg. Depth Ratio` = max(`Max. width`, na.rm = TRUE) / mean(Depth, na.rm = TRUE));
    combdata <- as.data.frame(ungroup(combdata));
    remove(temp);
    
    colnames(imgdata) <- c("Block", "Genotype", "Camera", "Median Number of Roots",
                           "Maximum Number of Roots", "Total Root Length", "Depth", "Maximum Width",
                           "Width-to-Depth Ratio", "Network Area", "Convex Area", "Solidity",
                           "Perimeter", "Average Radius", "Volume", "Surface Area", 
                           "Maximum Radius", "Lower Root Area", "Shallow Angle Frequency",
                           "Medium Angle Frequency", "Steep Angle Frequency", "Shallowness Index",
                           "Fine Radius Frequency", "Medium Radius Frequency", "Coarse Radius Frequency",
                           "Fineness Index", "Holes", "Computation");
    colnames(imgdataorig)[6:ncol(imgdataorig)] <- colnames(imgdata)[4:ncol(imgdata)];
}

###### Load manual data ######
manualdata <- read.csv(file="features_2016_mfeatures.csv", stringsAsFactors = FALSE, check.names=FALSE, na.strings = c("NA", "na"));
manualdata <- data.frame(lapply(manualdata[1:length(manualdata[[1]]),1:22], as.numeric), check.names=FALSE);
mdata <- manualdata[1:480, ];
mdata <- add_column(mdata, floor(mdata[[1]] / 1000), .after = 1);
colnames(mdata)[2] <- "Block";
mdata[[1]] <- (mdata[[1]] - ((mdata[[2]] - 1) * 1000));
mdata[[1]] <- factor(mdata[[1]]);
mdata[[2]] <- factor(mdata[[2]]);
mdata[[4]] <- factor(mdata[[4]]);

result <- mutate(rowwise(mdata), 
                 `Median Upper Angle` = median(c(`Angle 1 Adv/Up`, `Angle 2 Adv/Up`, `Angle 3 Adv/Up`, `Angle 4 Adv/Up`, `Angle 5 Adv/Up`), na.rm=TRUE),
                 `Mean Upper Angle` = mean(c(`Angle 1 Adv/Up`, `Angle 2 Adv/Up`, `Angle 3 Adv/Up`, `Angle 4 Adv/Up`, `Angle 5 Adv/Up`), na.rm=TRUE),
                 `Median Lower Angle` = median(c(`Angle 1 Lower`, `Angle 2 Lower`, `Angle 3 Lower`, `Angle 4 Lower`, `Angle 5 Lower`), na.rm=TRUE),
                 `Mean Lower Angle` = mean(c(`Angle 1 Lower`, `Angle 2 Lower`, `Angle 3 Lower`, `Angle 4 Lower`, `Angle 5 Lower`), na.rm=TRUE));
result[is.na(result)] <- NA;
result <- result[c(1:11, 24, 25, 17, 18, 26, 27)];

colnames(result)[6] <- "Taproot Diameter";
colnames(result)[7] <- "Overal Complexity Score";
colnames(result)[10] <- "Upper Primary Lateral Root Number";
colnames(result)[11] <- "Upper Secondary Lateral Root Density";
colnames(result)[13] <- "Upper Primary Lateral Angle Mean";
colnames(result)[12] <- "Upper Primary Lateral Angle Median";
colnames(result)[14] <- "Lower Primary Lateral Root Number";
colnames(result)[15] <- "Lower Secondary Lateral Root Density";
colnames(result)[17] <- "Lower Primary Lateral Angle Mean";
colnames(result)[16] <- "Lower Primary Lateral Angle Median";

result2 <- group_by(result, `Plot Number`, Block, Genotype) %>% summarise_all(funs(mean(., na.rm = TRUE)));
result2 <- result2[-4];
mdata <- result2;
mdata <- mdata[-1];
mdataorig <- result;
remove(result, result2);

##### Manual data loading complete #####

# Need to rename columns of manual data.
#mdata <- mdata[c(1,2,3,4,7,6,5,8,9,)]


# Initialize result data frame. Each column in the result contains a statistic that can be plotted.
statresults <- data.frame(cbind(colnames(imgdata)[4 : length(colnames(imgdata))]))
colnames(statresults)[1] <- "Features";

mstatresults <- data.frame(cbind(colnames(mdata)[3 : length(colnames(mdata))]))
colnames(mstatresults)[1] <- "Features";

cstatresults <- data.frame(cbind(colnames(combdata)[3 : length(colnames(combdata))]))
colnames(cstatresults)[1] <- "Features";

# To compute heritabilities
statresults <- compute_heritability(imgdata, statresults);
statresults <- compute_heritability(imgdata, statresults, camera = 1);
statresults <- compute_heritability(imgdata, statresults, camera = 2);
statresults <- compute_heritability(imgdata, statresults, camera = 3);
statresults <- compute_heritability(imgdata, statresults, camera = 4);
statresults <- compute_heritability(imgdata, statresults, camera = 5);
statresults <- compute_heritability(imgdata, statresults, 0, 1);

mdata <- add_column(mdata, rep(1, length(mdata[[1]])), .after = 2);
colnames(mdata)[3] <- "Camera";
mstatresults <- compute_heritability(mdata, mstatresults, camera = 1);
mdata <- mdata[-3];
colnames(mstatresults)[2] <- "Manual";

combdata <- add_column(combdata, rep(1, length(combdata[[1]])), .after = 2);
colnames(combdata)[3] <- "Camera";
cstatresults <- compute_heritability(combdata, cstatresults, camera = 1);
combdata <- combdata[-3];
colnames(cstatresults)[2] <- "Combined - New";

# To compute mean and SD of image and manual features.
astatresults <- data.frame();

astatresults <- compute_meanvars(imgdata, astatresults, camera = 1);
astatresults <- compute_meanvars(imgdata, astatresults, camera = 2);
astatresults <- compute_meanvars(imgdata, astatresults, camera = 3);
astatresults <- compute_meanvars(imgdata, astatresults, camera = 4);
astatresults <- compute_meanvars(imgdata, astatresults, camera = 5);
astatresults <- compute_meanvars(imgdata, astatresults);
#astatresults <- compute_meanvars(mdata, astatresults, 0, 1);
#astatresults <- compute_meanvars(combdata, astatresults, 0, 0, 1);

bstatresults <- data.frame();

bstatresults <- compute_meanvars_table(imgdata, bstatresults, camera = 1);
bstatresults <- compute_meanvars_table(imgdata, bstatresults, camera = 2);
bstatresults <- compute_meanvars_table(imgdata, bstatresults, camera = 3);
bstatresults <- compute_meanvars_table(imgdata, bstatresults, camera = 4);
bstatresults <- compute_meanvars_table(imgdata, bstatresults, camera = 5);
bstatresults <- compute_meanvars_table(imgdata, bstatresults);
bstatresults <- compute_meanvars_table(mdata, bstatresults, 0, 1);
bstatresults <- compute_meanvars_table(combdata, bstatresults, 0, 0, 1);
bstatresults$Mean=as.numeric(bstatresults$Mean);
bstatresults$SD=as.numeric(bstatresults$SD);
bstatresults$Min=as.numeric(bstatresults$Min);
bstatresults$Max=as.numeric(bstatresults$Max);

# Get Eta-squares for image features for plotting
imgetasquares <- compute_etasq(imgdata, as.character(statresults$Features)[order(statresults[[2]], decreasing = TRUE)]);

# Get results for plotting
results <- rbind(melt(statresults, id.vars = "Features"), melt(mstatresults, id.vars = "Features"), melt(cstatresults, id.vars = "Features"));
colnames(results)[2] <- "Camera";

# Get names for plotting
pnames <- c(as.character(statresults$Features), as.character(cstatresults$Features));
pnames <- c(pnames[order(c(statresults[[2]], cstatresults[[2]]), decreasing = TRUE)], 
            as.character(mstatresults$Features[order(mstatresults[[2]], decreasing = TRUE)]));

results <- rbind(results[26 : nrow(results), 1:ncol(results)], results[1 : 25, 1:ncol(results)]);
results$Features = factor(results$Features, levels = pnames, ordered = TRUE);
results$Camera = factor(results$Camera, levels = c("Left 2", "Left 1", "Center", "Right 1", "Right 2", "Combined", "Combined - New", "Manual", "Combined - Manova"), ordered = TRUE);

astatresults$Features = factor(astatresults$Features, levels = pnames, ordered = TRUE);
astatresults$Camera = factor(astatresults$Camera, levels = c("L2", "L1", "C", "R1", "R2", "AV", "CN", "M"), ordered = TRUE);
astatresults$Mean=as.numeric(astatresults$Mean);
astatresults$SD=as.numeric(astatresults$SD);

astatresults <- gather(astatresults, MeanOrSD, MSValue, 3:4) %>% spread(Features, MSValue);

############# Plot heritability #############
heritabilityplot <- ggplot(results) +    #geom_hline(yintercept =  1.3, linetype=2) +
    geom_segment(aes(x = Features, xend = Features, y = 0, yend=value), colour="grey") +
    geom_point(aes(x = Features, y = value, color=Camera, shape = Camera ), size=3) +
    ylab("Heritabilities") +
    xlab("Extracted Phenes") +
    scale_y_continuous(expand = c(0,0), limits = c(0, 1.0)) +
    #ggtitle("Heritabilities of digital and manual features.") +
    theme(axis.title.x = element_text(size=15, face="bold"), 
          axis.title.y = element_text(size=15, face="bold"), 
          axis.text.x = element_text(angle = 45, hjust = 1, size=14), 
          axis.text.y = element_text(size=14), 
          text=element_text(size=12),
          axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank(),
          plot.margin = unit(c(0.5,0.5,0.5,1), "cm"),
          plot.title = element_text(hjust = 0.5, size=14, face="bold"),
          legend.position = "top") +
    scale_colour_manual(values=c("darkolivegreen4", "darkolivegreen3", "firebrick", "deepskyblue", "dodgerblue3", "black", "black", "black", "black")) +
    scale_shape_manual(values=c(16,16,16,16,16,15,3,7,17)) +
    theme(legend.key = element_rect(fill = "transparent", colour = "transparent"));
########### Plot heritability end ###########

############# Plot means and variances #############
meanvarplots <- map(3:ncol(astatresults), function(index) {
    temp <- data.frame(as.character(astatresults[[1]]), astatresults[[2]], astatresults[[index]], 
                       stringsAsFactors = FALSE, check.names = FALSE);
    colnames(temp)[1] <- colnames(astatresults)[1];
    colnames(temp)[2] <- colnames(astatresults)[2];
    colnames(temp)[3] <- "FValue";
    
    temp$Camera <- factor(temp$Camera, levels = c("L2", "L1", "C", "R1", "R2", "AV", "CN", "M"), ordered = TRUE);
    temp2 <- spread(temp, MeanOrSD, FValue);
    colnames(temp2)[2] <- "Mean.FValue";
    colnames(temp2)[3] <- "SD.FValue";
    temp2 <- na.omit(temp2);
    
    meanvarplot <- ggplot(temp2) +    #geom_hline(yintercept =  1.3, linetype=2) +
    geom_linerange(aes(x = Camera, ymax = Mean.FValue + SD.FValue, ymin = Mean.FValue - SD.FValue), position = position_dodge(width = 0.9)) +
    geom_point(aes(x = Camera, y = Mean.FValue, shape = Camera), size=3) +
    ggtitle(colnames(astatresults)[index]) +
    #geom_segment(aes(x = Features, xend = Features, y = Mean - SD, yend=Mean + SD), colour=Camera) +
    #geom_point(aes(x = Features, y = Mean, color=Camera), size=3) +
    #ylab("Mean and SD") +
    #xlab("Extracted Phenes") +
    #scale_y_continuous(expand = c(0,0), limits = c(0, 1.0)) +
    #facet_wrap(~Features) +
    #ggtitle("Means and Standard Deviations of digital and manual features.") +
    theme(#axis.title.x = element_text(size=8, face="bold"),
          #axis.title.y = element_text(size=8, face="bold"),
          axis.title.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.x = element_text(size=7),
          axis.text.y = element_text(size=7),
          text=element_text(size=7),
          axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(),
          legend.position="none",
          panel.background = element_blank(),
          #plot.margin = unit(c(0.5,0.5,0.5,1), "cm"),
          plot.title = element_text(hjust = 0.5, size=7, face="bold")) +
          #legend.position = "top") +
    #scale_colour_manual(values=c("darkolivegreen4", "darkolivegreen3", "firebrick", "deepskyblue", "dodgerblue3", "black", "black", "black", "black")); # +
    scale_shape_manual(values=c(16,16,16,16,16,17,3,7,17));
    #theme(legend.key = element_rect(fill = "transparent", colour = "transparent"));
    
    # ggplot(glass2year.mese2, aes_string(x=names(glass2year.mese2)[index], y=names(glass2year.mese2)[index+18])) + 
    #     geom_smooth(method=lm, aes(colour=Treatment), alpha=.3) +
    #     geom_point(size=4, alpha = .8, aes(colour=Treatment)) + 
    #     scale_colour_manual(name = "Water", values=c("dodgerblue4", "firebrick4")) +
    #     elitetheme
    
    return(meanvarplot);
});

########### Plot means and variances ###########

############# Plot etasquares #############
imgetasquaresplot <- ggplot(imgetasquares) +
    geom_bar(aes(x = Features, y = Effectsize, fill=Factors), stat="identity") +
    geom_text(aes(x = Features, y = Cumeffectsize + (0.5 * Effectsize) + 0.001,label=paste0(round(Effectsize, 2) * 100, "")), size=6) +
    geom_text(aes(x = Features, y = Cumeffectsize + (0.5 * Effectsize) - 0.001,label=Significance), size=6) +
    scale_y_continuous(expand = c(0,0), limits = c(0, 1.01)) +
    ylab("Effect sizes (eta-squared)") +
    xlab("Phenes") +
    theme(axis.title.x = element_text(size=18, face="bold"), 
          axis.title.y = element_text(size=18, face="bold"), 
          axis.text.x = element_text(angle = 45, hjust = 1, size=18), 
          axis.text.y = element_text(size=18), 
          text=element_text(size=13),
          axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank(),
          plot.margin = unit(c(0.5,0.5,0.5,1), "cm"),
          plot.title = element_text(hjust = 0.5, size=14, face="bold"),
          legend.position = "top")
########### Plot etasquares end ###########

################### Plot correlations ###################
UseMeansForCamCorrelations = 0;
setwd("C:/Users/aseethepalli/Desktop/rscript");

#if (UseMeansForCamCorrelations == 0)
#{
    camcorr_nomean <- compute_camcor(imgdataorig);
    camcorr_nomean$feats <- factor(camcorr_nomean$feats, levels = pnames, ordered = TRUE);
#} else
#{
    # camcorr <- compute_camcor_with_means(imgdata);
    # camcorr$feats <- factor(camcorr$feats, levels = pnames, ordered = TRUE);
#}

cam1cor <- compute_cor(imgdataorig, mdataorig, 1);
cam2cor <- compute_cor(imgdataorig, mdataorig, 2);
cam3cor <- compute_cor(imgdataorig, mdataorig, 3);
cam4cor <- compute_cor(imgdataorig, mdataorig, 4);
cam5cor <- compute_cor(imgdataorig, mdataorig, 5);

camavgcor <- compute_cor(imgdataorig, mdataorig, 0);
camcoravg <- compute_cor(imgdataorig, mdataorig, 0,0);
camcormax <- compute_cor(imgdataorig, mdataorig, 0,0,0);
camcorwhichmax <- compute_cor(imgdataorig, mdataorig, 0,0,0,0);

cam1corp <- compute_corpval(imgdataorig, mdataorig, 1);
cam2corp <- compute_corpval(imgdataorig, mdataorig, 2);
cam3corp <- compute_corpval(imgdataorig, mdataorig, 3);
cam4corp <- compute_corpval(imgdataorig, mdataorig, 4);
cam5corp <- compute_corpval(imgdataorig, mdataorig, 5);
camavgcorp <- compute_corpval(imgdataorig, mdataorig, 0);
camcorpavg <- compute_corpval(imgdataorig, mdataorig, 0,0);
camcorpmax <- compute_corpval(imgdataorig, mdataorig, 0,0,0);

heatavgcor <- create_heatmap(camavgcor, "Correlation between average across cameras to manual");
heatcoravg <- create_heatmap(camcoravg, "Average correlation between individual cameras and manual");
heatcormax <- create_heatmap(camcormax, "Maximum correlation between individual cameras and manual");
heatcorwhichmax <- create_discrete_heatmap(camcorwhichmax, "Cameras correspoding to maximum correlation",
                                           c("darkolivegreen4", "darkolivegreen3", "firebrick", "deepskyblue", "dodgerblue3"),
                                           c("Left 2", "Left 1", "Center", "Right 1", "Right 2"), c("3", "1", "4", "2", "5")); #c("3", "1", "4", "2", "5"));

diffcoravg <- abs(camavgcor) - abs(camcoravg);
heatdiffcoravg <- create_heatmap(diffcoravg, "", 6, -0.15, 0.15);
diffcormax <- abs(camavgcor) - abs(camcormax);
heatdiffcormax <- create_heatmap(diffcormax, "", 6, -0.15, 0.15);

rmatmax <- ((diffcormax > 0) - 0.5) * 2;
maxpv <- camavgcorp;
maxpv[(camavgcorp < 0.05) & (camcorpmax < 0.05)] <- 1;
maxpv[!((camavgcorp < 0.05) & (camcorpmax < 0.05))] <- 0;
rmatmax <- rmatmax * maxpv;
remove(maxpv);

rmatavg <- ((diffcoravg > 0) - 0.5) * 2;
avgpv <- camavgcorp;
avgpv[(camavgcorp < 0.05) & (camcorpavg < 0.05)] <- 1;
avgpv[!((camavgcorp < 0.05) & (camcorpavg < 0.05))] <- 0;
rmatavg <- rmatavg * avgpv;
remove(avgpv);

diffcoravgpval <- create_discrete_heatmap(rmatavg, "", c("#4e4eac", "white", "#a14242"), 
                                          c("Better", "Insignificant", "Worse"), c("1","0","-1"));
diffcormaxpval <- create_discrete_heatmap(rmatmax, "", c("#4e4eac", "white", "#a14242"), 
                                          c("Better", "Insignificant", "Worse"), c("1","0","-1"));

# camcorrplot <- ggplot(camcorr, aes(x = feats)) +    
#     geom_errorbar(aes(ymax = means + sds, ymin = means - sds), position = "dodge") +
#     geom_point(aes(x = feats, y = means), size=3) +
#     ylab("Pearson correlation coefficient.") +
#     xlab("Extracted phenes") +
#     ggtitle("Mean inter-camera correlations and standard deviations for each digital trait.") +
#     theme(axis.title.x = element_text(size=15, face="bold"), 
#           axis.title.y = element_text(size=15, face="bold"), 
#           axis.text.x = element_text(angle = 45, hjust = 1, size=14), 
#           axis.text.y = element_text(size=14), 
#           text=element_text(size=12),
#           axis.line = element_line(colour = "black"),
#           panel.grid.major = element_blank(),
#           panel.grid.minor = element_blank(),
#           panel.border = element_blank(),
#           panel.background = element_blank(),
#           plot.margin = unit(c(0.5,0.5,0.5,1), "cm"),
#           plot.title = element_text(hjust = 0.5, size=14, face="bold"),
#           legend.position = "right") +
#     scale_colour_manual(values=c("darkolivegreen4", "darkolivegreen3", "firebrick", "deepskyblue", "dodgerblue3", "black", "black", "black", "black")) +
#     theme(legend.key = element_rect(fill = "transparent", colour = "transparent"));

camcorrplot_nomean <- ggplot(camcorr_nomean, aes(x = feats)) +    
    geom_errorbar(aes(ymax = means + sds, ymin = means - sds), position = "dodge") +
    geom_point(aes(x = feats, y = means), size=3) +
    ylab("Pearson correlation coefficient.") +
    xlab("Extracted phenes") +
    #ggtitle("Mean inter-camera correlations and standard deviations for each digital trait.") +
    theme(axis.title.x = element_text(size=18, face="bold"), 
          axis.title.y = element_text(size=18, face="bold"), 
          axis.text.x = element_text(angle = 45, hjust = 1, size=18), 
          axis.text.y = element_text(size=18), 
          text=element_text(size=12),
          axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank(),
          plot.margin = unit(c(0.5,0.5,0.5,1), "cm"),
          plot.title = element_text(hjust = 0.5, size=14, face="bold"),
          legend.position = "right") +
    scale_colour_manual(values=c("darkolivegreen4", "darkolivegreen3", "firebrick", "deepskyblue", "dodgerblue3", "black", "black", "black", "black")) +
    theme(legend.key = element_rect(fill = "transparent", colour = "transparent"));

################# Plot correlations end #################



########## Do real plotting ###########
plotwidth <- 11;
plotheight <- 7.5;

pdf(file = "heritability_combined_HIS_ROLL_mean_effectsizes.pdf", width = plotwidth, height = plotheight, family = "ArialMT", useDingbats=FALSE);
multiplot(heritabilityplot, cols=1);
dev.off();

pdf(file = "correlations_combined_HIS_ROLL_mean.pdf", width = plotwidth * 2 - 6, height = plotheight * 2 + 1, family = "ArialMT", useDingbats=FALSE);
multiplot(heatavgcor, heatcormax, heatcoravg, heatcorwhichmax, cols=2);
dev.off();

pdf(file = "correlations_max_diff_combined_HIS_ROLL_mean.pdf", width = plotwidth * 2 - 6, height = plotheight + 0.5, family = "ArialMT", useDingbats=FALSE);
multiplot(heatdiffcormax, diffcormaxpval, cols=2);
dev.off();

pdf(file = "correlations_avg_diff_combined_HIS_ROLL_mean.pdf", width = plotwidth * 2 - 6, height = plotheight + 0.5, family = "ArialMT", useDingbats=FALSE);
multiplot(heatdiffcoravg, diffcoravgpval, cols=2);
dev.off();

#if (UseMeansForCamCorrelations == 0)
#{
pdf(file = "correlations_camera_HIS_ROLL_no_mean.pdf", width = plotwidth + 2.5, height = plotheight, family = "ArialMT", useDingbats=FALSE);
multiplot(camcorrplot_nomean, cols=1);
dev.off();

#} else
#{
# pdf(file = "correlations_camera_HIS_ROLL_mean.pdf", width = plotwidth, height = plotheight, family = "ArialMT", useDingbats=FALSE);
# multiplot(camcorrplot, cols=1);
# dev.off();

#}

pdf(file = "meanvars_HIS_ROLL_mean.pdf", width = 8, height = 8, family = "ArialMT", useDingbats=FALSE);
do.call(function(...) {grid.arrange(ncol=5, ...)}, meanvarplots)
dev.off();

pdf(file = "effectsizes_HIS_ROLL_mean.pdf", width = plotwidth, height = plotheight, family = "ArialMT", useDingbats=FALSE);
multiplot(imgetasquaresplot, cols=1);
dev.off();

######## Do real plotting end #########

########## Output CSV ###########
write.csv(bstatresults, file = "features_MeanSDMinMax.csv", row.names = FALSE);
######## Output CSV end #########


