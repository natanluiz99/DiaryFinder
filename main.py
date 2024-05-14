from fastapi import  Form, FastAPI, Request, HTTPException, Query, Path, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF

from typing import List, Optional
import urllib.request
import requests
import os
import subprocess


app = FastAPI()

# Configurar o diretório de arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



# CORS (Cross-Origin Resource Sharing) Middleware para permitir solicitações de diferentes origens
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Altere isso para permitir apenas origens específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Inicializar o pipeline BERT para question answering
qa_pipeline = pipeline("question-answering", model="mrm8488/bert-base-portuguese-cased-finetuned-squad-v1-pt")



# Diretório para salvar os arquivos enviados
UPLOAD_DIRECTORY = "uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

@app.get("/download")
async def download_page(request: Request):
    return templates.TemplateResponse("download.html", {"request": request})

@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...), question: str = Form(...)):
    try:
        # Ler o conteúdo do arquivo PDF enviado
        contents = await file.read()

        # Extrair texto do PDF
        pdf_document = fitz.open(stream=contents)
        text = ""
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            text += page.get_text()

        # Dividir o texto em segmentos menores (por exemplo, frases)
        segments = text.split('. ')
        
        # Inicializar uma lista para armazenar as respostas
        answers = []
        
        # Fazer uma chamada ao modelo para cada segmento
        for segment in segments:
            # Verificar se o segmento contém conteúdo suficiente
            if segment.strip():
                answer = qa_pipeline(question=question, context=segment)
                answers.append(answer['answer'])
        
        # Combinar as respostas em um único texto
        combined_answer = ' '.join(answers)

        # Retornar a resposta como parte do JSON
        return JSONResponse(content={"question": question, "answer": combined_answer})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erro ao processar o arquivo PDF: {str(e)}"})



#-------------------------------------------------------------
# Endpoint para receber a análise do PDF e exibi-la na página de download
@app.post("/display-analysis", response_class=HTMLResponse)
async def display_analysis(file: UploadFile = File(...), question: str = Form(...)):
    try:
        # Ler o conteúdo do arquivo PDF enviado
        contents = await file.read()

        # Extrair texto do PDF
        pdf_document = fitz.open(stream=contents)
        text = ""
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            text += page.get_text()

        # Dividir o texto em segmentos menores (por exemplo, frases)
        segments = text.split('. ')
        
        # Inicializar uma lista para armazenar as respostas
        answers = []
        
        # Fazer uma chamada ao modelo para cada segmento
        for segment in segments:
            # Verificar se o segmento contém conteúdo suficiente
            if segment.strip():
                answer = qa_pipeline(question=question, context=segment)
                answers.append(answer['answer'])
        
        # Combinar as respostas em um único texto
        combined_answer = ' '.join(answers)

        # Renderizar o resultado na página de download
        return templates.TemplateResponse("download.html", {"request": request, "result": combined_answer})

    
    except Exception as e:
        # Se ocorrer um erro, retorne um JSON com uma mensagem de erro
        return {"error": f"Erro ao processar o arquivo PDF: {str(e)}"}
#-------------------------------------------------------------




@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Salvar o arquivo no diretório de upload
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        return {"detail": f"Arquivo {file.filename} enviado com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao fazer upload do arquivo: {str(e)}")


@app.get("/files/{file_name}")
async def get_file(file_name: str):
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/octet-stream", filename=file_name)
    raise HTTPException(status_code=404, detail="Arquivo não encontrado")


@app.get("/analyze-text")
async def analyze_text(text: str):
    try:
        # Fazer a pergunta ao modelo BERT
        question = "Qual o nome do prefeito dessa cidade citada?"
        answer = qa_pipeline(question=question, context=text)

        return {"question": question, "answer": answer["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao analisar o texto: {str(e)}")







def merge_pdfs(pdf_files, output_pdf):
    merger = PdfMerger()

    try:
        # Adiciona cada arquivo PDF ao merger
        for pdf_file in pdf_files:
            merger.append(pdf_file)

        # Salva o arquivo PDF de saída
        merger.write(output_pdf)
        merger.close()

        return {"message": f"PDFs mesclados com sucesso em {output_pdf}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao mesclar os PDFs: {str(e)}")













# Rota para mesclar PDFs
@app.post("/merge-pdfs")
async def merge_pdfs(files: List[UploadFile] = File(...)):
    try:
        merger = PdfMerger()

        for uploaded_file in files:
            # Salva o arquivo temporariamente
            file_path = f"temp/{uploaded_file.filename}"
            with open(file_path, "wb") as file_object:
                file_object.write(uploaded_file.file.read())
            
            # Adiciona o PDF ao objeto de mesclagem
            merger.append(file_path)

        output_pdf_path = "merged_pdf.pdf"
        merger.write(output_pdf_path)
        merger.close()

        # Remove os arquivos temporários
        for uploaded_file in files:
            file_path = f"temp/{uploaded_file.filename}"
            os.remove(file_path)

        return {"detail": f"PDFs mesclados com sucesso. Arquivo salvo em {output_pdf_path}"}
    except Exception as e:
        return {"detail": f"Erro ao mesclar os PDFs: {e}"}


def baixar_arquivo_pdf(url: str, file_name: str, download_dir: str):
    try:
        # Verifica se o diretório de download existe, se não, cria-o
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        file_path = os.path.join(download_dir, file_name)

        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'MyApp/1.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, file_path)
        return {"message": f"Download do arquivo {file_name} concluído com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao baixar o arquivo: {str(e)}")

# Rota para baixar arquivos PDF a partir de URLs em um arquivo de texto
@app.post("/download-from-txt")
async def download_from_txt(file: UploadFile = File(...)):
    try:
        # Lê o conteúdo do arquivo de texto
        contents = await file.read()

        # Decodifica o conteúdo em linhas e remove espaços em branco
        lines = contents.decode().split('\n')
        urls = [line.strip() for line in lines if line.strip() and "files" not in line.lower()]  # Ignora linhas com "files"

        # Diretório de download
        download_dir = os.path.join(os.path.dirname(__file__), 'pdfs_diarios')

        # Itera sobre cada URL e baixa o arquivo PDF
        for i, url in enumerate(urls):
            file_name = f"diario_oficial_{i}.pdf"  # Nome do arquivo com base no número da linha
            baixar_arquivo_pdf(url, file_name, download_dir)

        return {"message": f"Downloads de {len(urls)} arquivos concluídos."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o arquivo de texto: {str(e)}")







@app.get("/download")
async def download_page(request: Request):
    return templates.TemplateResponse("download.html", {"request": request})

# Função para executar o scrapy
def executar_scrapy(cidade: str, dataInicio: str, dataFim: str):
    comando = f"cd querido-diario/data_collection && scrapy crawl {cidade} -a start_date={dataInicio} -a end_date={dataFim} -s LOG_FILE=log_{cidade}.txt -o {cidade}.csv"
    resultado_comando = subprocess.run(comando, shell=True, capture_output=True, text=True)

    # Verificar se a execução do comando foi bem-sucedida
    if resultado_comando.returncode == 0:
        mensagem_aviso = f"Scrapy para {cidade} concluído com sucesso!"
    else:
        mensagem_aviso = f"Erro ao executar o scrapy para {cidade}."

    return mensagem_aviso

# Rota para executar o scrapy
@app.get("/executar-scrapy/{cidade}/{dataInicio}/{dataFim}")
def executar_scrapy_endpoint(request: Request, cidade: str, dataInicio: str, dataFim: str):
    mensagem_aviso = executar_scrapy(cidade, dataInicio, dataFim)
    return JSONResponse(content={"mensagem_aviso": mensagem_aviso})


# Rota para listar os arquivos CSV disponíveis
@app.get("/abrir-csv")
def abrir_csv_endpoint(request: Request):
    # Lista todos os arquivos CSV na pasta data_collection
    csv_files = [filename.split('.')[0] for filename in os.listdir("querido-diario/data_collection") if filename.endswith(".csv")]

    return templates.TemplateResponse("selecionar_csv.html", {"request": request, "csv_files": csv_files})



def trocar_colunas(data_rows):
    for row in data_rows:
        row[0], row[8] = row[8], row[0]



@app.get("/exibir-csv/{cidade}")
def exibir_csv_endpoint(request: Request, cidade: str):
    try:
        # Ler o contdeo do arquivo CSV
        with open(f"querido-diario/data_collection/{cidade}.csv", "r") as file:
            # Ler as linhas do arquivo
            csv_lines = file.readlines()
        
        # Obter o cabeçalho (primeira linha)
        header = csv_lines[0].strip().split(',')

        # Processar as linhas de dados (excluindo o cabeçalho)
        data_rows = []
        download_links = []  # Lista para armazenar os links de download

        for line in csv_lines[1:]:
            # Verificar se a linha contém colchetes '[' e ']'
            if '[' in line and ']' in line:
                # Encontrar os índices dos colchetes '[' e ']'
                start_index = line.index('[')
                end_index = line.index(']')

                # Extrair a parte da linha dentro dos colchetes
                list_values = line[start_index + 1 : end_index]

                # Remover a parte dentro dos colchetes da linha original
                line = line.replace('[' + list_values + ']', '')

                # Dividir a linha por vírgulas
                columns = line.split(',')

                # Adicionar a parte da linha antes dos colchetes como uma coluna
                data_row = [segment.strip() for segment in columns[0].split(' ', maxsplit=1) if segment.strip()]

                # Adicionar os valores da lista dentro dos colchetes como uma coluna
                data_row.append(list_values)

                # Adicionar a parte da linha após os colchetes como uma coluna
                data_row.extend(columns[1:])

                # Adicionar a linha processada
                data_rows.append(data_row)
            else:
                # Se não houver colchetes, dividir normalmente por vírgulas
                data_row = [segment.strip() for segment in line.split(',')]

                # Mover os números entre a segunda e a terceira vírgula para a posição correta
                if len(data_row) >= 4:
                    data_row[2] = f"{data_row[2]}, {data_row[3]}"
                    data_row.pop(3)  # Remover o elemento na posição 3 após a mudança

                # Adicionar a linha processada
                data_rows.append(data_row)

             # Adicionar o link de download à lista
            download_links.append(data_row[7])  # Assumindo que o link de download está na coluna 7

        # Trocar o conteúdo das colunas 0 e 8
        trocar_colunas(data_rows)

        # Renderizar o template HTML com a tabela
        return templates.TemplateResponse("exibir_csv.html", {"request": request, "header": header, "data_rows": data_rows, "download_links": download_links})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Arquivo CSV não encontrado")




# Rota para lidar com o download de todos os links

@app.get("/atualizar-repositorio")
def atualizar_repositorio_endpoint(request: Request):
    resultados_comandos = atualizar_repositorio()
    mensagem_aviso = "Scrapy concluído."  # Adicione a mensagem padrão ou deixe em branco se preferir
    return templates.TemplateResponse("index.html", {"request": request, "resultados_comandos": resultados_comandos, "mensagem_aviso": mensagem_aviso})

def atualizar_repositorio():
    # Comandos para configurar o ambiente virtual e instalar as dependências
    comandos = [
        "pip install pyyaml==6.0.1",
        "git clone https://github.com/okfn-brasil/querido-diario.git",
        "python -m venv venv_querido",
        ".\\venv_querido\\Scripts\\activate",  # Use .\\ para ativar o ambiente virtual no Windows
        "pip install -r querido-diario/data_collection/requirements-dev.txt --no-deps",
        "cd querido-diario/data_collection && scrapy list",
    
    ]

# Rota para listar as cidades disponíveis
@app.get("/listar-cidades")
def listar_cidades():
    # Comando para listar as cidades disponíveis usando scrapy
    resultado_comando = subprocess.run("cd querido-diario/data_collection && scrapy list", shell=True, capture_output=True, text=True)
    # Verificar se o comando foi bem-sucedido
    if resultado_comando.returncode == 0:
        # Obter a lista de cidades
        cidades = resultado_comando.stdout.splitlines()
        return JSONResponse(content={"cidades": cidades})
    else:
        raise HTTPException(status_code=500, detail="Erro ao listar as cidades disponíveis")
    
