# Distribuidor de Escalas para Residentes
Esse software é destinado a criar e distribuir escalas para residentes médicos da maneira mais eficiente e balanceada possível. Originalmente desenvolvido para residentes do Instituto de Medicina Integral Professor Fernando Figueira (IMIP).

<img width="902" alt="gui" src="https://github.com/brenerrr/distribuidor_escalas_residencia/assets/36827826/91dc2123-2f1f-4410-8f79-d0940420192c">

Versões executáveis para Windows podem ser baixadas na [seção de Releases](https://github.com/brenerrr/distribuidor_escalas_residencia/tags). 

# Instalação 

Instale os módulos necessários listados no arquivo _requirements.txt_ (de preferência num _virtual environment_ dedicado). **Utilize a versão de Python 3.10**. 

```python
pip install -r requirements.txt
```

# Execução

Para executar a GUI, utilize o comando 

```python
python main.py
```

Adicione todas as áreas e pessoas pertinentes, além de possíveis folgas e restrições a quantidade máximas de turnos na aba _Inputs_. Feito isso, associe cada pessoa à sua respectiva área na aba _Tabela Pessoas-Áreas_. 
Finalmente, adicione todos os turnos devidos na aba _Turnos_. 

Parar gerar a planilha com a escala, clique em _Gerar escalas_ e aguarde a finalização do programa. 

Note que um arquivo executável _standlone_ pode ser gerado a partir do comando abaixo. Note que é necessário desabilitar seu antivírus ao gerar esse executável. 

```python
pyinstaller app.spec
```

# Exemplo

Renomeie o arquivo _sample_inputs.json_ para _inputs.json_ e execute o programa para ter uma noção de como ele deve ser preenchido. 

# TODOs 

- Adicionar tool tips explanatórias
- Encerrar _Worker Thread_ caso a GUI seja fechada. 
- Corrigir bug que pode acontecer quando alguém que trabalha em turnos não obrigatórios é remanejado.
- Adicionar testes mais detalhados para o _Manager_
