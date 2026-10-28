import React from "react";
import { AlertTriangleIcon, CheckCircleIcon, ErrorIcon, InboxIcon, InfoIcon, SpinnerIcon } from "./icons.jsx";

/**
 * Selo de classificação: SEMPRE cor + texto + ícone, nunca cor isolada.
 * `label` já vem traduzido pelo chamador ("Maligno"/"Benigno").
 */
export function ResultBadge({ label, kind, className = "" }) {
  const Icon = kind === "malignant" ? AlertTriangleIcon : CheckCircleIcon;
  const classe = kind === "malignant" ? "badge-malignant" : "badge-benign";
  return (
    <span className={`badge ${classe} ${className}`.trim()}>
      <Icon className="badge-icon" />
      {label}
    </span>
  );
}

/** Bloco de estado para carregamento, vazio ou erro — usado nas 4 telas. */
export function StateBlock({ kind = "loading", title, children }) {
  const ICONES = { loading: SpinnerIcon, empty: InboxIcon, error: ErrorIcon };
  const Icon = ICONES[kind] || InfoIcon;
  const papel = kind === "error" ? { role: "alert" } : { role: "status" };
  return (
    <div className="state-block" {...papel}>
      <Icon className="state-icon" />
      {title && <h3>{title}</h3>}
      {children && <p className="text-muted">{children}</p>}
    </div>
  );
}

/** Alerta de card (erro, aviso, informação ou sucesso) com ícone + texto. */
export function Alert({ tone = "info", title, children, ...rest }) {
  const ICONES = { error: ErrorIcon, warning: AlertTriangleIcon, info: InfoIcon, success: CheckCircleIcon };
  const Icon = ICONES[tone] || InfoIcon;
  const papel = tone === "error" || tone === "warning" ? "alert" : "status";
  return (
    <div className={`alert alert-${tone}`} role={papel} {...rest}>
      <Icon className="alert-icon" />
      <div className="alert-body">
        {title && <strong className="alert-title">{title}</strong>}
        {children}
      </div>
    </div>
  );
}

/** Linhas de esqueleto para estado de carregamento dentro de um cartão. */
export function SkeletonLines({ count = 3 }) {
  return (
    <div aria-hidden="true">
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="skeleton skeleton-line" style={{ width: `${92 - i * 14}%` }} />
      ))}
    </div>
  );
}
