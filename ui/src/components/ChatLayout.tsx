import { useState } from 'react'
import { deleteThread } from '../lib/api'
import { loadTrace, renameSession } from '../lib/sessions'
import type { TraceEntry } from '../lib/types'
import { useSession } from '../hooks/useSession'
import { useStream } from '../hooks/useStream'
import { ChatInput } from './ChatInput'
import { HITLCard } from './HITLCard'
import { MessageList } from './MessageList'
import { Sidebar } from './Sidebar'

export function ChatLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const {
    sessions,
    currentThreadId,
    trace,
    settings,
    setSettings,
    switchSession,
    newSession,
    removeSession,
    appendTrace,
    truncateAndReplaceTrace,
    refreshSessions,
    autoTitleIfNew,
    getCurrentCheckpointId,
  } = useSession()

  const { isStreaming, isHITLPaused, hitlPayload, activeNode, liveReasoning, runStream, clearHITL } = useStream()

  const handleSend = async (text: string) => {
    const checkpointId = await getCurrentCheckpointId()
    const userEntry: TraceEntry = { role: 'user', content: text, checkpointId }
    appendTrace(userEntry)
    autoTitleIfNew(text)

    await runStream(
      currentThreadId,
      {
        message: text,
        monthly_income: settings.monthlyIncome,
        current_date: new Date().toISOString().split('T')[0],
        use_sample_data: settings.useSampleData,
      },
      (entry) => appendTrace(entry),
    )
    refreshSessions()
  }

  const handleApprove = async (message?: string) => {
    const userEntry: TraceEntry = {
      role: 'user',
      content: message ? message : '✅ Approved — continuing.',
    }
    appendTrace(userEntry)
    clearHITL()

    await runStream(
      currentThreadId,
      { resume: { confirmed: true, message: message ?? null } },
      (entry) => appendTrace(entry),
    )
    refreshSessions()
  }

  const handleReject = async () => {
    appendTrace({ role: 'user', content: '❌ Rejected — continuing.' })
    clearHITL()

    await runStream(
      currentThreadId,
      { resume: { confirmed: false } },
      (entry) => appendTrace(entry),
    )
    refreshSessions()
  }

  const handleEditSave = async (index: number, newText: string) => {
    const currentTrace = loadTrace(currentThreadId)
    const entry = currentTrace[index]
    if (!entry || entry.role !== 'user') return

    const checkpointId = entry.checkpointId ?? null

    if (checkpointId === null) {
      await deleteThread(currentThreadId)
    }

    const newEntry: TraceEntry = { role: 'user', content: newText, checkpointId }
    truncateAndReplaceTrace(index, newEntry)

    if (index === 0) autoTitleIfNew(newText)

    await runStream(
      currentThreadId,
      {
        message: newText,
        monthly_income: settings.monthlyIncome,
        current_date: new Date().toISOString().split('T')[0],
        use_sample_data: settings.useSampleData,
        checkpoint_id: checkpointId,
      },
      (agentEntry) => appendTrace(agentEntry),
    )
    refreshSessions()
  }

  return (
    <div className="flex h-screen bg-white overflow-hidden">
      {/* Sidebar toggle button */}
      <button
        onClick={() => setSidebarOpen((v) => !v)}
        className="fixed top-3 left-3 z-30 w-8 h-8 flex items-center justify-center rounded-lg text-gray-500 hover:bg-gray-100 transition-colors"
        title={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
      >
        {sidebarOpen ? (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        )}
      </button>

      {/* Sidebar — floating overlay */}
      {sidebarOpen && (
        <>
          {/* Backdrop for mobile / click-outside */}
          <div
            className="fixed inset-0 z-10 sm:hidden"
            onClick={() => setSidebarOpen(false)}
          />
          <div className="fixed left-4 top-4 bottom-4 z-20 rounded-xl shadow-[0_0_20px_rgba(0,0,0,0.08)] border border-gray-100 bg-white overflow-hidden"
               style={{ width: '260px' }}>
            <Sidebar
              sessions={sessions}
              currentThreadId={currentThreadId}
              settings={settings}
              onSessionClick={switchSession}
              onNewSession={newSession}
              onDeleteSession={removeSession}
              onRenameSession={(id, title) => {
                renameSession(id, title)
                refreshSessions()
              }}
              onSettingsChange={setSettings}
            />
          </div>
        </>
      )}

      {/* Main chat area */}
      <div
        className="flex flex-col flex-1 min-w-0 transition-all duration-200"
        style={{ marginLeft: sidebarOpen ? '284px' : '0' }}
      >
        <div className="flex flex-col flex-1 overflow-hidden">
          <MessageList
            trace={trace}
            isStreaming={isStreaming}
            activeNode={activeNode}
            liveReasoning={liveReasoning}
            onEditSave={handleEditSave}
            hitlCard={
              isHITLPaused && hitlPayload ? (
                <HITLCard
                  payload={hitlPayload}
                  onApprove={handleApprove}
                  onReject={handleReject}
                  disabled={isStreaming}
                />
              ) : null
            }
          />

          <ChatInput onSend={handleSend} disabled={isStreaming || isHITLPaused} />
        </div>
      </div>
    </div>
  )
}
