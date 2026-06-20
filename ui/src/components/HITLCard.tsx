import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Components } from 'react-markdown'
import type { PendingConfirmation } from '../lib/types'

// Table components styled for the blue summary box
const summaryComponents: Components = {
  table: ({ children }) => (
    <div className="overflow-x-auto my-2">
      <table className="min-w-full text-xs border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-blue-100">{children}</thead>,
  th: ({ children }) => (
    <th className="px-3 py-1.5 text-left font-semibold border border-blue-200 text-blue-800 whitespace-nowrap">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="px-3 py-1.5 border border-blue-200 text-blue-900">{children}</td>
  ),
  tr: ({ children, ...props }) => (
    <tr className="even:bg-blue-50/60" {...(props as object)}>{children}</tr>
  ),
}

// Table components styled for the detail list items (neutral)
const detailComponents: Components = {
  table: ({ children }) => (
    <div className="overflow-x-auto my-1">
      <table className="min-w-full text-xs border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-gray-100">{children}</thead>,
  th: ({ children }) => (
    <th className="px-2 py-1 text-left font-semibold border border-gray-200 text-gray-700 whitespace-nowrap">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="px-2 py-1 border border-gray-200 text-gray-600">{children}</td>
  ),
  tr: ({ children, ...props }) => (
    <tr className="even:bg-gray-50" {...(props as object)}>{children}</tr>
  ),
}

interface HITLCardProps {
  payload: PendingConfirmation
  onApprove: (message?: string) => void
  onReject: () => void
  disabled?: boolean
}

export function HITLCard({ payload, onApprove, onReject, disabled }: HITLCardProps) {
  const [showDetails, setShowDetails] = useState(false)
  const [clarifyText, setClarifyText] = useState('')
  const [mode, setMode] = useState<'default' | 'clarify'>('default')

  return (
    <div className="mx-auto max-w-xl my-4">
      <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
        <div className="px-4 py-3 bg-amber-50 border-b border-amber-100 flex items-center gap-2">
          <span className="text-base">✋</span>
          <span className="font-semibold text-amber-800 text-sm">Confirmation Required</span>
        </div>

        <div className="px-4 py-3 space-y-3">
          <div className="flex gap-4 text-xs text-gray-500">
            <span><span className="font-medium text-gray-700">Agent:</span> {payload.agent}</span>
            <span><span className="font-medium text-gray-700">Action:</span> {payload.action}</span>
          </div>

          <div className="bg-blue-50 border border-blue-100 rounded-lg px-3 py-2 text-sm text-blue-900
            [&_p]:my-0.5 [&_ul]:my-1 [&_ul]:pl-4 [&_li]:my-0.5 [&_strong]:font-semibold [&_code]:bg-blue-100 [&_code]:px-1 [&_code]:rounded">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={summaryComponents}>{payload.summary}</ReactMarkdown>
          </div>

          {payload.details && payload.details.length > 0 && (
            <div>
              <button
                onClick={() => setShowDetails((v) => !v)}
                className="text-xs text-gray-500 hover:text-gray-700 flex items-center gap-1"
              >
                <span>{showDetails ? '▾' : '▸'}</span>
                <span>{showDetails ? 'Hide details' : 'Show details'}</span>
              </button>
              {showDetails && (
                <ul className="mt-2 space-y-1.5">
                  {payload.details.map((d, i) => (
                    <li key={i} className="text-xs text-gray-600
                      [&_p]:my-0 [&_p]:inline [&_strong]:font-semibold [&_code]:bg-gray-100 [&_code]:px-0.5 [&_code]:rounded">
                      <span className="text-gray-300 mr-2">•</span>
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={detailComponents}>{d}</ReactMarkdown>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {mode === 'clarify' && (
            <textarea
              className="w-full p-2 text-sm border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-gray-300"
              rows={2}
              placeholder="Add a clarification message…"
              value={clarifyText}
              onChange={(e) => setClarifyText(e.target.value)}
              autoFocus
            />
          )}
        </div>

        <div className="px-4 py-3 border-t border-gray-100 flex gap-2 flex-wrap">
          {mode === 'default' ? (
            <>
              <button
                onClick={() => onApprove()}
                disabled={disabled}
                className="flex items-center gap-1.5 px-4 py-2 text-sm bg-gray-800 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50"
              >
                <span>✅</span> Approve
              </button>
              <button
                onClick={() => setMode('clarify')}
                disabled={disabled}
                className="flex items-center gap-1.5 px-4 py-2 text-sm border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 disabled:opacity-50"
              >
                <span>💬</span> Clarify
              </button>
              <button
                onClick={onReject}
                disabled={disabled}
                className="flex items-center gap-1.5 px-4 py-2 text-sm border border-red-200 text-red-600 rounded-lg hover:bg-red-50 disabled:opacity-50"
              >
                <span>❌</span> Reject
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => onApprove(clarifyText.trim() || undefined)}
                disabled={disabled}
                className="flex items-center gap-1.5 px-4 py-2 text-sm bg-gray-800 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50"
              >
                Send
              </button>
              <button
                onClick={() => { setMode('default'); setClarifyText('') }}
                className="px-4 py-2 text-sm text-gray-500 hover:text-gray-700"
              >
                Cancel
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
