import { useCallback, useEffect, useRef, useState } from 'react'
import { deleteThread, getThreadState } from '../lib/api'
import {
  autoTitle,
  createSession,
  deleteSession,
  listSessions,
  loadTrace,
  renameSession,
  saveTrace,
  touchSession,
} from '../lib/sessions'
import type { Session, TraceEntry } from '../lib/types'

interface UseSessionResult {
  sessions: Session[]
  currentThreadId: string
  trace: TraceEntry[]
  settings: { monthlyIncome: number; useSampleData: boolean }
  setSettings: (s: { monthlyIncome: number; useSampleData: boolean }) => void
  switchSession: (threadId: string) => void
  newSession: () => void
  removeSession: (threadId: string) => void
  renameCurrentSession: (title: string) => void
  appendTrace: (entry: TraceEntry) => void
  truncateAndReplaceTrace: (upToIndex: number, newEntry: TraceEntry) => TraceEntry[]
  persistTrace: (trace: TraceEntry[]) => void
  refreshSessions: () => void
  autoTitleIfNew: (firstMsg: string) => void
  getCurrentCheckpointId: () => Promise<string | null>
}

export function useSession(): UseSessionResult {
  const [sessions, setSessions] = useState<Session[]>(() => listSessions())
  const [currentThreadId, setCurrentThreadId] = useState<string>(() => {
    const existing = listSessions()
    if (existing.length > 0) return existing[0].threadId
    const session = createSession('New chat')
    return session.threadId
  })
  const [trace, setTrace] = useState<TraceEntry[]>(() => {
    const existing = listSessions()
    if (existing.length > 0) return loadTrace(existing[0].threadId)
    return []
  })
  const [settings, setSettings] = useState({ monthlyIncome: 3200.0, useSampleData: true })

  const currentThreadIdRef = useRef(currentThreadId)
  currentThreadIdRef.current = currentThreadId

  const refreshSessions = useCallback(() => {
    setSessions(listSessions())
  }, [])

  const switchSession = useCallback((threadId: string) => {
    setCurrentThreadId(threadId)
    setTrace(loadTrace(threadId))
    currentThreadIdRef.current = threadId
  }, [])

  const newSession = useCallback(() => {
    const session = createSession('New chat')
    setSessions(listSessions())
    setCurrentThreadId(session.threadId)
    setTrace([])
    currentThreadIdRef.current = session.threadId
  }, [])

  const removeSession = useCallback(
    (threadId: string) => {
      deleteThread(threadId).catch(() => {})
      deleteSession(threadId)
      const remaining = listSessions()
      setSessions(remaining)
      if (threadId === currentThreadIdRef.current) {
        if (remaining.length > 0) {
          switchSession(remaining[0].threadId)
        } else {
          const session = createSession('New chat')
          setSessions(listSessions())
          setCurrentThreadId(session.threadId)
          setTrace([])
          currentThreadIdRef.current = session.threadId
        }
      }
    },
    [switchSession],
  )

  const renameCurrentSession = useCallback(
    (title: string) => {
      renameSession(currentThreadIdRef.current, title)
      refreshSessions()
    },
    [refreshSessions],
  )

  const persistTrace = useCallback((t: TraceEntry[]) => {
    saveTrace(currentThreadIdRef.current, t)
    touchSession(currentThreadIdRef.current)
  }, [])

  const appendTrace = useCallback(
    (entry: TraceEntry) => {
      setTrace((prev) => {
        const next = [...prev, entry]
        saveTrace(currentThreadIdRef.current, next)
        touchSession(currentThreadIdRef.current)
        return next
      })
      refreshSessions()
    },
    [refreshSessions],
  )

  const truncateAndReplaceTrace = useCallback(
    (upToIndex: number, newEntry: TraceEntry): TraceEntry[] => {
      const trimmed = [...loadTrace(currentThreadIdRef.current).slice(0, upToIndex), newEntry]
      setTrace(trimmed)
      saveTrace(currentThreadIdRef.current, trimmed)
      touchSession(currentThreadIdRef.current)
      return trimmed
    },
    [],
  )

  const autoTitleIfNew = useCallback(
    (firstMsg: string) => {
      const tid = currentThreadIdRef.current
      const session = listSessions().find((s) => s.threadId === tid)
      if (session && session.title === 'New chat') {
        renameSession(tid, autoTitle(firstMsg))
        refreshSessions()
      }
    },
    [refreshSessions],
  )

  const getCurrentCheckpointId = useCallback(async (): Promise<string | null> => {
    try {
      const state = await getThreadState(currentThreadIdRef.current)
      return state.checkpoint_id
    } catch {
      return null
    }
  }, [])

  useEffect(() => {
    refreshSessions()
  }, [refreshSessions])

  return {
    sessions,
    currentThreadId,
    trace,
    settings,
    setSettings,
    switchSession,
    newSession,
    removeSession,
    renameCurrentSession,
    appendTrace,
    truncateAndReplaceTrace,
    persistTrace,
    refreshSessions,
    autoTitleIfNew,
    getCurrentCheckpointId,
  }
}
