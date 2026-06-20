import { useRef, useState } from 'react'
import type { Session } from '../lib/types'

interface SidebarProps {
  sessions: Session[]
  currentThreadId: string
  settings: { monthlyIncome: number; useSampleData: boolean }
  onSessionClick: (threadId: string) => void
  onNewSession: () => void
  onDeleteSession: (threadId: string) => void
  onRenameSession: (threadId: string, title: string) => void
  onSettingsChange: (settings: { monthlyIncome: number; useSampleData: boolean }) => void
}

function relativeTime(ts: number): string {
  const delta = (Date.now() - ts) / 1000
  if (delta < 60) return 'just now'
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`
  return new Date(ts).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })
}

export function Sidebar({
  sessions,
  currentThreadId,
  settings,
  onSessionClick,
  onNewSession,
  onDeleteSession,
  onRenameSession,
  onSettingsChange,
}: SidebarProps) {
  const [menuOpenFor, setMenuOpenFor] = useState<string | null>(null)
  const [renamingFor, setRenamingFor] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const renameInputRef = useRef<HTMLInputElement>(null)

  const startRename = (threadId: string, currentTitle: string) => {
    setRenamingFor(threadId)
    setRenameValue(currentTitle)
    setMenuOpenFor(null)
    setTimeout(() => renameInputRef.current?.focus(), 0)
  }

  const commitRename = () => {
    if (renamingFor && renameValue.trim()) {
      onRenameSession(renamingFor, renameValue.trim())
    }
    setRenamingFor(null)
  }

  return (
    <div className="flex flex-col h-full w-64">
      {/* Header */}
      <div className="px-3 pt-3 pb-2 flex items-center justify-between">
        <span className="font-semibold text-sm text-gray-700">SmartFin</span>
        <button
          onClick={onNewSession}
          className="w-7 h-7 flex items-center justify-center rounded-lg text-gray-500 hover:bg-gray-100 transition-colors"
          title="New chat"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 5v14m-7-7h14" />
          </svg>
        </button>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto px-2 space-y-0.5">
        {sessions.length === 0 && (
          <p className="text-xs text-gray-400 px-2 py-3 text-center">No conversations yet</p>
        )}
        {sessions.map((s) => (
          <div
            key={s.threadId}
            className={`group relative flex items-center rounded-lg px-2 py-2 cursor-pointer transition-colors ${
              s.threadId === currentThreadId ? 'bg-gray-100' : 'hover:bg-gray-50'
            }`}
            onClick={() => {
              if (renamingFor !== s.threadId && menuOpenFor !== s.threadId) {
                onSessionClick(s.threadId)
              }
            }}
          >
            {renamingFor === s.threadId ? (
              <input
                ref={renameInputRef}
                className="flex-1 text-sm bg-white border border-gray-300 rounded px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-gray-300"
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                onBlur={commitRename}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commitRename()
                  if (e.key === 'Escape') setRenamingFor(null)
                }}
                onClick={(e) => e.stopPropagation()}
              />
            ) : (
              <>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-800 truncate">{s.title}</p>
                  <p className="text-xs text-gray-400">{relativeTime(s.lastActivityAt)}</p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    setMenuOpenFor(menuOpenFor === s.threadId ? null : s.threadId)
                  }}
                  className="opacity-0 group-hover:opacity-100 ml-1 w-6 h-6 flex items-center justify-center rounded text-gray-400 hover:text-gray-600 hover:bg-gray-200 transition-all flex-shrink-0"
                >
                  ⋯
                </button>
              </>
            )}

            {menuOpenFor === s.threadId && (
              <div
                className="absolute right-2 top-8 z-30 bg-white border border-gray-200 rounded-lg shadow-lg py-1 min-w-[120px]"
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  className="w-full text-left px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
                  onClick={() => startRename(s.threadId, s.title)}
                >
                  Rename
                </button>
                <button
                  className="w-full text-left px-3 py-1.5 text-sm text-red-600 hover:bg-red-50"
                  onClick={() => {
                    setMenuOpenFor(null)
                    onDeleteSession(s.threadId)
                  }}
                >
                  Delete
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Settings */}
      <div className="border-t border-gray-100 px-2 py-2">
        <button
          onClick={() => setShowSettings((v) => !v)}
          className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-gray-500 hover:bg-gray-50 transition-colors text-sm"
        >
          <span>⚙️</span>
          <span>Settings</span>
          <span className="ml-auto">{showSettings ? '▾' : '▸'}</span>
        </button>
        {showSettings && (
          <div className="mt-2 px-2 space-y-3 pb-1">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Monthly Income (SGD)</label>
              <input
                type="number"
                className="w-full border border-gray-200 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-gray-300"
                value={settings.monthlyIncome}
                onChange={(e) =>
                  onSettingsChange({ ...settings, monthlyIncome: parseFloat(e.target.value) || 0 })
                }
              />
            </div>
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="useSample"
                checked={settings.useSampleData}
                onChange={(e) => onSettingsChange({ ...settings, useSampleData: e.target.checked })}
                className="rounded"
              />
              <label htmlFor="useSample" className="text-xs text-gray-600">
                Use sample transactions
              </label>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
