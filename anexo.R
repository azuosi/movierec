library("ggplot2")
library("recommenderlab")
library("caTools") #ROC curve, logitboost, gif...
library("dplyr")
# Teste Estudo de Caso - Algoritmo de Recomendação 
# ETAPA 1- Importação dos dados + manipulação da tabela ----
#importação via Enviroment>Import Dataset> From Text (base)> /Caminhodoarquivo> Import
data = u.data
# renomeando colunas
names(data) <- c("user_id", "item_id", "rating")
# retirando a variável timestamp, pois não é necessária
data <- data[ ,-c(4)]
str(data) #converte em string
data_mat <- as(data, "realRatingMatrix") #lista, formato exigido pelo pacote de recomendação
#histograma da distribuição das classificações
ggplot(data = data, aes(x = rating)) +
  geom_histogram(bins = 10, fill="purple")
# hist_norm <- hist(getRatings(normalize(data_mat, method="Z-score")), breaks=10)
#histograma filmes mais vistos segundo a classificação.
ggplot(data = data, aes(x = item_id)) +
  geom_histogram(bins = 10, fill="purple")
# Explorando a similaridade
sim_usuarios <- similarity(data_mat[1:4, ], 
                           method = "cosine", 
                           which = "user.id")
as.matrix(sim_usuarios)
image(as.matrix(sim_usuarios), main = "User similarity")
# ETAPA 2 - Dimensão dos dados ----
Number_Ratings <-  nrow(data)
Number_Movies <-  length(unique(data$item_id))
Number_Users <-  length(unique(data$user_id))
# ETAPA 3 - Separação dados + aplicação das técnicas ----
# Usa conjunto de treino
# Conjunto de treino/teste = 80/20
which_train <- sample(x = c(TRUE, FALSE), 
                      size = nrow(data),
                      replace = TRUE, 
                      prob = c(0.7, 0.3))
# Abaixo se definem 2 vezes cada conjunto, uma como data frame e outra como matrix.
# Pois as funções só funcionam com matriz e os gráficos com data frame.

data_train <- data[which_train, ]
data_train_mat <- as(data_train, "realRatingMatrix")
data_test <- data[!which_train, ]
data_test_mat <- as(data_test, "realRatingMatrix")

# Aplicação do algoritmo

#Com base na popularidade
rec <- Recommender(data_mat[1:700], method = "POPULAR")

#Aleatória
rec1 <- Recommender(data_mat[1:700], method = "RANDOM")

#Com base na rede de colaboração 
rec2 <- Recommender(data_mat[1:700], method = "UBCF")
# ETAPA 04 -  Lista de top 5 recomendações para usuários com base em cada modelo proposto ----
#obtem top 5 recommended items model rec
rec.items <- predict(rec, data_mat[701:943, ], n=5)
as(rec.items, "list")  # to display them
#classificações
rec.items.rat <- predict(rec, data_mat[701:943, ], type="ratings")
as(rec.items.rat, "list") # Predict the ratings. Para tdas as obs
as(rec.items.rat, "matrix")[,1:10] #Predict the ratings for the first 10 items  
#obtem top 5 recommended items model rec1
rec.items1 <- predict(rec1, data_mat[701:943, ], n=5)
as(rec.items1, "list") # to display them
#Classificações do Rec1 
rec.items.rat.1 <- predict(rec1, data_mat[701:943, ], type="ratings")
as(rec.items.rat.1, "matrix")[,1:10]
#obtem top 5 recommended items model rec2
rec.items2<- predict(rec2, data_mat[701:943, ], n=5)
as(rec.items2, "list") # to display them
#Classificações do Rec2
rec.items.rat.2 <- predict(rec2, data_mat[701:943, ], type="ratings")
as(rec.items.rat.2, "matrix")[,1:10]
#ETAPA 5- Avaliação: MSE ---- 
e <- evaluationScheme(data_mat, method = "cross-validation",train=0.7, k=5, given=5, goodRating = 4)
algorithms <- list("random items" = list(name="RANDOM", param=NULL), 
                   "popular items" = list(name="POPULAR", param="cosine"), 
                   "user-based CF" = list(name="UBCF", param="pearson")
                   
                   results <- evaluate(e, algorithms, type="topNList", n=c(2, 7, 10, 15, 20, 25))
                   #ETAPA 06- Plotting ROC curve ---- 
                   plot(results, annotate=c(1), legend="topleft")
                   title("ROC curve")
                   plot(results, "prec/rec", annotate=3, legend="bottomright")
                   title("Precision-recall")  
                   
                   
                   
                   