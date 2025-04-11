> rainfall <- c(799,1174.8,865.1,1334.6,635.4,918.5,685.5,784.2,985,882.8,1071)
> rainfall.timeseries <- ts(rainfall,start = c(2021,1),frequency = 12) 
> print(rainfall.timeseries)
        Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov
2021  799.0 1174.8  865.1 1334.6  635.4  918.5  685.5  784.2  985.0  882.8 1071.0
> png(file = "rainfall.png")
> plot(rainfall.timeseries) 
> dev.off()
null device 
          1 
> plot(rainfall.timeseries) 
