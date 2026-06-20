type Update = Record<string, unknown>

function latestAiText(update: Update): string | null {
  const msgs = (update.messages as unknown[]) || []
  for (let i = msgs.length - 1; i >= 0; i--) {
    const msg = msgs[i] as Record<string, unknown>
    if (msg?.type === 'ai') {
      const content = msg.content
      if (typeof content === 'string' && content.trim()) return content
    }
  }
  return null
}

export function summarizeUpdate(node: string, update: Update): string {
  if (node === 'supervisor') {
    const ai = latestAiText(update)
    if (ai) return ai
    const active = update.active_agent as string | null
    const pending = update.pending_intent as string | null
    if (!active || active === 'end') return 'All planned work complete — ending graph.'
    const parts = [`Dispatching → \`${active}\``]
    if (pending && pending !== active)
      parts.push(`Will follow up with \`${pending}\` after prep work completes.`)
    return parts.join('\n\n')
  }

  if (node === 'expense_analysis') {
    const cats = (update.categorised_transactions as unknown[]) || []
    const trends = (update.spending_trends as unknown[]) || []
    const pending = update.pending_confirmation as Record<string, unknown> | null
    const parts = [`Categorised **${cats.length}** transactions across **${trends.length}** categories.`]
    if (pending) parts.push(`⏸ Awaiting confirmation — _${pending.summary}_`)
    return parts.join('\n\n')
  }

  if (node === 'budget_planning') {
    const pending = update.pending_confirmation as Record<string, unknown> | null
    if (pending && typeof pending.action === 'string' && pending.action.startsWith('clarify'))
      return `⏸ Needs clarification — _${pending.summary}_`
    const allocs = (update.budget_allocations as unknown[]) || []
    const warnings = (update.budget_warnings as unknown[]) || []
    const summary = (update.budget_summary as string) || ''
    const lines = [`**${allocs.length}** budget allocation(s), **${warnings.length}** warning(s).`]
    if (summary) lines.push(`_${summary}_`)
    const ai = latestAiText(update)
    if (ai) lines.push(ai)
    return lines.join('\n\n')
  }

  if (node === 'goal_planning') {
    const pending = update.pending_confirmation as Record<string, unknown> | null
    const goals = (update.goals as unknown[]) || []
    const action = (pending?.action as string) || ''

    if (action === 'approve_goal_planning') {
      // New goal is awaiting HITL confirmation — keep trace entry brief;
      // full summary is shown in the HITL card below.
      return `⏸ New goal pending your confirmation. Tracking **${goals.length}** goal(s) total.`
    }

    if (action.startsWith('clarify')) {
      const parts = [`⏸ Needs clarification — _${pending?.summary}_`]
      for (const d of (pending?.details as string[]) || []) parts.push(`- ${d}`)
      return parts.join('\n')
    }

    // For queries, missing-field prompts, and post-confirmation evaluation:
    // use the LLM's actual response text.
    const ai = latestAiText(update)
    if (ai) return ai

    return `Tracking **${goals.length}** goal(s).`
  }

  if (node === 'anomaly_detection') {
    const flags = (update.anomaly_flags as unknown[]) || []
    const explanation = (update.anomaly_explanation as string) || ''
    const parts = [`Scanned transactions — **${flags.length}** anomaly flag(s).`]
    if (explanation) parts.push(explanation)
    return parts.join('\n\n')
  }

  if (node === 'health_assessment') {
    const ai = latestAiText(update)
    if (ai) return ai
    const hs = update.health_summary as Record<string, unknown> | null
    if (!hs) return '_(no health summary produced)_'
    const rating = String(hs.rating ?? '?').toUpperCase()
    const dti = ((hs.debt_to_income_ratio as number) ?? 0) * 100
    const reserves = (hs.liquid_reserve_months as number) ?? 0
    const lines = [
      `**Rating:** \`${rating}\``,
      `**DTI:** ${dti.toFixed(0)}%  ·  **Reserves:** ${reserves.toFixed(1)} months`,
    ]
    if (hs.income_concentration_risk) lines.push('⚠️ Income concentration risk')
    if (hs.sustained_overspending) lines.push('⚠️ Sustained overspending')
    const observations = (hs.observations as string[]) || []
    if (observations.length) {
      lines.push('**Observations:**')
      for (const obs of observations) lines.push(`- ${obs}`)
    }
    const alerts = (update.alerts as unknown[]) || []
    if (alerts.length) lines.push(`**Alerts:** ${alerts.length} total`)
    return lines.join('\n')
  }

  if (node === 'confirm') return 'Confirmation processed → handing back to supervisor.'

  return '_(no output)_'
}
