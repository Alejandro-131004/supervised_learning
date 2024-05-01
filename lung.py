import pandas as pd
from io import StringIO

# Dados CSV como uma string para simular a leitura de um arquivo.
csv_data = """Name,Surname,Age,Smokes,AreaQ,Alkhol,Result
John,Wick,35,3,5,4,1
John,Constantine,27,20,2,5,1
Camela,Anderson,30,0,5,2,0
Alex,Telles,28,0,8,1,0
Diego,Maradona,68,4,5,6,1
Cristiano,Ronaldo,34,0,10,0,0
Mihail,Tal,58,15,10,0,0
Kathy,Bates,22,12,5,2,0
Nicole,Kidman,45,2,6,0,0
Ray,Milland,52,18,4,5,1
Fredric,March,33,4,8,0,0
Yul,Brynner,18,10,6,3,0
Joan,Crawford,25,2,5,1,0
Jane,Wyman,28,20,2,8,1
Anna,Magnani,34,25,4,8,1
Katharine,Hepburn,39,18,8,1,0
Katharine,Hepburn,42,22,3,5,1
Barbra,Streisand,19,12,8,0,0
Maggie,Smith,62,5,4,3,1
Glenda,Jackson,73,10,7,6,1
Jane,Fonda,55,15,1,3,1
Maximilian,Schell,33,8,8,1,0
Gregory,Peck,22,20,6,2,0
Sidney,Poitier,44,5,8,1,0
Rex,Harrison,77,3,2,6,1
Lee,Marvin,21,20,5,3,0
Paul,Scofield,37,15,6,2,0
Rod,Steiger,34,12,8,0,0
John,Wayne,55,20,1,4,1
Gene,Hackman,40,20,2,7,1
Marlon,Brando,36,13,5,2,0
Jack,Lemmon,56,20,3,3,1
Jack,Nicholson,47,15,1,8,1
Peter,Finch,62,25,3,4,1
Richard,Dreyfuss,26,10,7,2,0
Dustin,Hoffman,25,20,8,2,0
Henry,Henry,59,20,3,4,1
Robert,Duvall,62,15,5,5,1
Ellen,Burstyn,33,25,8,2,0
Faye,Dunaway,37,10,5,3,0
Diane,Keaton,50,20,2,4,1
Jane,Fonda,47,12,8,0,0
Sally,Field,69,20,5,4,1
Sissy,Spacek,63,20,4,5,1
Jessica,Lange,39,15,7,2,0
Gwyneth,Paltrow,21,20,8,3,0
Halle,Berry,31,20,9,4,0
Nicole,Kidman,28,10,4,1,0
Charlize,Theron,53,20,6,3,1
Katharine,Hepburn,62,20,5,6,1
Katharine,Hepburn,42,12,6,2,0
Barbra,Streisand,44,30,1,6,1
Maggie,Smith,26,34,1,8,1
Glenda,Jackson,35,20,5,1,0
Ernest,Borgnine,26,13,6,1,0
Alec,Guinness,77,20,5,4,1
Charlton,Heston,75,15,3,5,1
Gregory,Peck,43,30,3,8,1
Sidney,Poitier,51,25,9,0,0"""

# Lendo os dados para um DataFrame
data = pd.read_csv(StringIO(csv_data))

# Verificando os nomes das colunas
print("Nomes das colunas:", data.columns)

# Ajustando o nome da coluna "Result" para remover espaços extras, se necessário
data.columns = data.columns.str.strip()

# Contando '1's e '0's
count_1 = (data['Result'] == 1).sum()
count_0 = (data['Result'] == 0).sum()

# Calculando porcentagens
total = len(data)
percentage_1 = (count_1 / total) * 100
percentage_0 = (count_0 / total) * 100

print(f"Quantidade de '1': {count_1}, Porcentagem de '1': {percentage_1:.2f}%")
print(f"Quantidade de '0': {count_0}, Porcentagem de '0': {percentage_0:.2f}%")
