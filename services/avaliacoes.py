"""
Registro e consulta do histórico de avaliações (US07 / UC07).
"""
from sqlalchemy.orm import joinedload

from db.models import Avaliacao, Papel, Usuario, db
from services.modelos import modelo_registrado_mais_recente


def registrar(*, caracteristicas: dict, resultado: dict, tempo_ms: float,
              identificado: bool, nome: str | None, prontuario: str | None,
              medico_id: int | None) -> Avaliacao:
    """Grava a avaliação em uma única transação."""
    modelo = modelo_registrado_mais_recente(resultado["modelo_utilizado"])
    av = Avaliacao(
        medico_id=medico_id,
        modelo_id=modelo.id if modelo else None,
        caracteristicas=caracteristicas,
        classificacao=resultado["classificacao"],
        probabilidade_maligno=resultado["probabilidade_maligno"],
        confianca=resultado["confianca"],
        faixa_confianca=resultado["faixa_confianca"],
        limiar_utilizado=resultado["limiar_utilizado"],
        tempo_resposta_ms=round(tempo_ms, 2),
        identificado=identificado,
        paciente_nome=nome,
        paciente_prontuario=prontuario,
    )
    db.session.add(av)
    db.session.commit()
    return av


def medico_existe(medico_id: int) -> bool:
    u = db.session.get(Usuario, medico_id)
    return u is not None and u.ativo and u.papel is Papel.MEDICO


def paciente_dados(av: Avaliacao):
    """Dados do paciente para exibição, ou None se o registro é anônimo."""
    if not av.identificado:
        return None
    return {"nome": av.paciente_nome, "prontuario": av.paciente_prontuario}


def listar(filtros: dict) -> dict:
    q = Avaliacao.query.options(joinedload(Avaliacao.modelo))
    if filtros["classificacao"]:
        q = q.filter(Avaliacao.classificacao == filtros["classificacao"])
    if filtros["identificado"] is not None:
        q = q.filter(Avaliacao.identificado.is_(filtros["identificado"]))
    if filtros["inicio"]:
        q = q.filter(Avaliacao.criado_em >= filtros["inicio"])
    if filtros["fim_exclusivo"]:
        q = q.filter(Avaliacao.criado_em < filtros["fim_exclusivo"])
    if filtros["confianca_min"] is not None:
        q = q.filter(Avaliacao.confianca >= filtros["confianca_min"])
    if filtros["confianca_max"] is not None:
        q = q.filter(Avaliacao.confianca <= filtros["confianca_max"])

    q = q.order_by(Avaliacao.criado_em.desc(), Avaliacao.id.desc())
    total = q.count()
    limite, pagina = filtros["limite"], filtros["pagina"]
    itens = q.offset((pagina - 1) * limite).limit(limite).all()

    return {
        "itens": [a.to_dict(paciente=paciente_dados(a)) for a in itens],
        "paginacao": {
            "pagina": pagina,
            "limite": limite,
            "total": total,
            "paginas": max(1, -(-total // limite)),
        },
    }


def obter(avaliacao_id: int):
    av = db.session.get(Avaliacao, avaliacao_id)
    return None if av is None else av.to_dict(paciente=paciente_dados(av))
