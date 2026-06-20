import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Components } from 'react-markdown'
import type { TraceEntry } from '../lib/types'

const tableComponents: Components = {
  table: ({ children }) => (
    <div className="overflow-x-auto my-2">
      <table className="min-w-full text-sm border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-gray-100">{children}</thead>,
  th: ({ children }) => (
    <th className="px-3 py-2 text-left font-semibold border border-gray-200 text-gray-700 whitespace-nowrap">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="px-3 py-2 border border-gray-200 text-gray-700">{children}</td>
  ),
  tr: ({ children, ...props }) => (
    <tr className="even:bg-gray-50" {...(props as object)}>{children}</tr>
  ),
}

export const AGENT_META: Record<string, [string, string]> = {
  supervisor:        ['🧭', 'Supervisor'],
  expense_analysis:  ['📊', 'Expense Analysis'],
  budget_planning:   ['💰', 'Budget Planning'],
  goal_planning:     ['🎯', 'Goal Planning'],
  anomaly_detection: ['🚨', 'Anomaly Detection'],
  health_assessment: ['❤️', 'Health Assessment'],
  confirm:           ['✋', 'HITL Confirm'],
}

interface MessageBubbleProps {
  entry: TraceEntry
  index: number
  onEditSave?: (index: number, newText: string) => void
}

export function MessageBubble({ entry, index, onEditSave }: MessageBubbleProps) {
  const [editing, setEditing] = useState(false)
  const [editText, setEditText] = useState(entry.content)
  const [reasoningOpen, setReasoningOpen] = useState(false)

  if (entry.role === 'user') {
    return (
      <div className="flex justify-end mb-4 group">
        <div className="relative max-w-[70%]">
          {editing ? (
            <div className="flex flex-col gap-2">
              <textarea
                className="w-full min-w-[300px] p-3 rounded-lg border border-gray-300 text-sm resize-y focus:outline-none focus:ring-2 focus:ring-gray-300"
                rows={3}
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                autoFocus
              />
              <div className="flex gap-2 justify-end">
                <button
                  onClick={() => setEditing(false)}
                  className="px-3 py-1 text-sm text-gray-500 hover:text-gray-700"
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    if (editText.trim() && onEditSave) {
                      onEditSave(index, editText.trim())
                      setEditing(false)
                    }
                  }}
                  className="px-3 py-1 text-sm bg-gray-800 text-white rounded-lg hover:bg-gray-700"
                >
                  Save & Resend
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className="bg-gray-100 text-black px-4 py-3 rounded-lg text-sm leading-relaxed whitespace-pre-wrap">
                {entry.content}
              </div>
              {onEditSave && (
                <button
                  onClick={() => { setEditText(entry.content); setEditing(true) }}
                  className="absolute -left-8 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 hover:text-gray-600 text-xs p-1"
                  title="Edit message"
                >
                  ✏️
                </button>
              )}
            </>
          )}
        </div>
      </div>
    )
  }

  // Supervisor routing decision — shown in subdued gray, no collapse needed.
  if (entry.isReasoning) {
    return (
      <div className="flex flex-col mb-3 pl-3 border-l-2 border-gray-100">
        <span className="text-[10px] text-gray-300 uppercase tracking-wider mb-0.5 font-medium">
          Routing
        </span>
        <div className="text-xs text-gray-400 leading-relaxed [&_code]:bg-transparent [&_code]:text-gray-400">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={tableComponents}>{entry.content}</ReactMarkdown>
        </div>
      </div>
    )
  }

  // Regular agent message (with optional collapsible reasoning from ReAct loop).
  const [emoji, name] = AGENT_META[entry.agent ?? ''] ?? ['🤖', entry.agent ?? 'Agent']
  const reasoningSteps = entry.reasoning ? entry.reasoning.split('\n\n') : []

  return (
    <div className="flex flex-col mb-6">
      <div className="flex items-center gap-1.5 mb-1.5">
        <span className="text-sm">{emoji}</span>
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">{name}</span>
      </div>

      {reasoningSteps.length > 0 && (
        <div className="mb-2">
          <button
            onClick={() => setReasoningOpen((v) => !v)}
            className="flex items-center gap-1 text-[11px] text-gray-400 hover:text-gray-500 transition-colors mb-1 select-none"
          >
            <svg
              viewBox="0 0 12 12"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              className={`w-2.5 h-2.5 transition-transform ${reasoningOpen ? 'rotate-90' : ''}`}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 2l4 4-4 4" />
            </svg>
            {reasoningOpen ? 'Hide' : 'Show'} thinking ({reasoningSteps.length} step{reasoningSteps.length > 1 ? 's' : ''})
          </button>

          {reasoningOpen && (
            <div className="mt-1.5 border border-gray-100 rounded-xl overflow-hidden">
              <div className="px-3 py-2.5 space-y-2.5 bg-gray-50">
                {reasoningSteps.filter((s) => s.trim()).map((step, i) => (
                  <div key={i} className="flex gap-2.5">
                    <span className="text-[10px] font-mono text-gray-300 mt-0.5 flex-shrink-0 w-5 text-right">
                      {i + 1}
                    </span>
                    <div className="text-xs text-gray-500 leading-relaxed
                      [&_p]:my-0.5 [&_ul]:my-1 [&_ul]:pl-4 [&_li]:my-0.5
                      [&_strong]:font-medium [&_code]:bg-gray-200 [&_code]:px-0.5 [&_code]:rounded">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={tableComponents}>{step.trim()}</ReactMarkdown>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="text-sm text-gray-800 leading-relaxed
        [&_p]:my-1.5 [&_ul]:my-1.5 [&_ul]:pl-5 [&_ol]:my-1.5 [&_ol]:pl-5
        [&_li]:my-0.5 [&_h1]:text-base [&_h1]:font-bold [&_h1]:mt-3 [&_h1]:mb-1
        [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:mt-2.5 [&_h2]:mb-1
        [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:mt-2 [&_h3]:mb-0.5
        [&_strong]:font-semibold [&_em]:italic
        [&_code]:bg-gray-100 [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-xs [&_code]:font-mono
        [&_pre]:bg-gray-100 [&_pre]:rounded-lg [&_pre]:p-3 [&_pre]:overflow-x-auto [&_pre]:my-2
        [&_blockquote]:border-l-2 [&_blockquote]:border-gray-300 [&_blockquote]:pl-3 [&_blockquote]:text-gray-500 [&_blockquote]:my-2
        [&_hr]:border-gray-200 [&_hr]:my-3">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={tableComponents}>{entry.content}</ReactMarkdown>
      </div>
    </div>
  )
}
