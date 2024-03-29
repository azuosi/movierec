library("ggplot2")
library("recommenderlab")
library("caTools") #ROC curve, logitboost, gif...
library("dplyr")

# Teste Estudo de Caso - Algoritmo de Recomenda��o 

# ETAPA 1- Importa��o dos dados + manipula��o da tabela ----


#importa��o via Enviroment>Import Dataset> From Text (base)> /Caminhodoarquivo> Import
u.data <- read.delim("~/DOUTORADO/aprendizagem estatistica/Trabalho recomendacao/u.data.txt")
data = u.data

#importa��o dos nomes dos filmes
movies <- read.csv("movies.csv",stringsAsFactors=FALSE)



# renomeando colunas
names(data) <- c("user_id", "item_id", "rating")

 
# retirando a vari�vel timestamp, pois n�o � necess�ria
data <- data[ ,-c(4)]

# Tentando Pegar os nomes e g�neros dos filmes de movies e trazer pra data

data <- left_join(data, movies %>% select(movieId, title, genres),
                  by = c("item_id" = "movieId"))

#renomeando
names(data)[4:5] <- c("title", "genres")

str(data) #converte em string
data_mat <- as(data, "realRatingMatrix") #lista, formato exigido pelo pacote de recomenda��o



#histograma da distribui��o das classifica��es/ratings
ggplot(data = data, aes(x = rating)) +
  geom_histogram(bins = 10, fill="purple")




# ETAPA 2 - Dimens�o dos dados ----


Number_Ratings <-  nrow(data)
Number_Movies <-  length(unique(data$item_id))
Number_Users <-  length(unique(data$user_id))

  
  
# ETAPA 3 - Separa��o dados + aplica��o das t�cnicas ----
 # Usa conjunto de treino

# Conjunto de treino/teste = 80/20

which_train <- sample(x = c(TRUE, FALSE), 
                      size = nrow(data),
                      replace = TRUE, 
                      prob = c(0.7, 0.3))

# Abaixo se definem 2 vezes cada conjunto, uma como data frame e outra como matrix.
# Pois as fun��es s� funcionam com matriz e os gr�ficos com data frame.

data_train <- data[which_train, ]
data_train_mat <- as(data_train, "realRatingMatrix")
data_test <- data[!which_train, ]
data_test_mat <- as(data_test, "realRatingMatrix")


  
# Aplica��o do algoritmo

  
#Com base na popularidade
    rec <- Recommender(data_mat[1:700], method = "POPULAR")
                     
   
#Aleat�ria
    rec1 <- Recommender(data_mat[1:700], method = "RANDOM")
    
#Com base na rede de colabora��o 
    rec2 <- Recommender(data_mat[1:700], method = "UBCF")

    
# ETAPA 04 -  Lista de top 5 recomenda��es para usu�rios com base em cada modelo proposto ----
  
    
    #obtem top 5 recommended items model rec
    rec.items <- predict(rec, data_mat[701:943, ], n=5)
    as(rec.items, "list")  # to display them
    rec.list_mov <- as.data.frame(as(rec.items, "list"))[ ,1:10] #Col= usu�rio e raws = recomenda��es
  
    
     #classifica��es
    rec.items.rat <- predict(rec, data_mat[701:943, ], type="ratings")
    as(rec.items.rat, "list") # Predict the ratings. Para tdas as obs
    as(rec.items.rat, "matrix")[,1:10] #Predict the ratings for the first 10 items
    
    
    #obtem top 5 recommended items model rec1
    rec.items1 <- predict(rec1, data_mat[701:943, ], n=5)
    as(rec.items1, "list") # to display them
    rec.list_mov1 <- as.data.frame(as(rec.items1, "list"))[ ,1:10]
   
    
     #Classifica��es do Rec1 
    rec.items.rat.1 <- predict(rec1, data_mat[701:943, ], type="ratings")
    as(rec.items.rat.1, "matrix")[,1:10]
    
    
    
    #obtem top 5 recommended items model rec2
    rec.items2<- predict(rec2, data_mat[701:943, ], n=5)
    as(rec.items2, "list") # to display them
    rec.list_mov2 <-as.data.frame(as(rec.items2, "list"))[,1:10] # to display them
   
    #Classifica��es do Rec2
    rec.items.rat.2 <- predict(rec2, data_mat[701:943, ], type="ratings")
    as(rec.items.rat.2, "matrix")[,1:10]
    
    
  
    
    
#ETAPA 5- Avalia��o: MSE ---- 
    
    e <- evaluationScheme(data_mat, method = "cross-validation",train=0.7, k=5, given=5, goodRating = 4)
    
    algorithms <- list("random items" = list(name="RANDOM", param=NULL), 
                       "popular items" = list(name="POPULAR", param="cosine"), 
                       "user-based CF" = list(name="UBCF", param="pearson")
                       )
    
    results <- evaluate(e, algorithms, type="topNList", n=c(2, 7, 10, 15, 20, 25))
    
    
    
#ETAPA 06- Plotting ROC curve ---- 
   
    
    plot(results, annotate=c(1), legend="topleft")
    title("ROC curve")
    plot(results, "prec/rec", annotate=3, legend="bottomright")
    title("Precision-recall")  
    
    
 