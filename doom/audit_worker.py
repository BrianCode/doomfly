"""Audit hashing and logging off the simulation thread.

The simulation thread submits (event, light, frame, counts) and continues; one
worker computes the three SHA-256 digests, completes the event in place and
writes the audit line. hashlib releases the GIL for these buffers, so hashing
overlaps the next tic. Lines keep submission order (one FIFO, one consumer).
A full queue blocks the producer, counted in stats(), and never drops a line.
"""
import hashlib,json,queue,threading
DIGESTS=('input_sha256','source_frame_sha256','spike_counts_sha256')

class AuditHandle:
    """Resolves to the completed event once its audit line is written."""
    __slots__=('event','done','error')
    def __init__(self,event):self.event=event;self.done=threading.Event();self.error=None
    def result(self,timeout=None):
        if not self.done.wait(timeout):raise TimeoutError('audit digests not ready')
        if self.error:raise RuntimeError(self.error)
        return self.event

class AuditWorker:
    def __init__(self,logger,maxsize=32):
        self.logger=logger;self.queue=queue.Queue(maxsize)
        self.submitted=0;self.written=0;self.blocked=0;self.failed=None
        self._thread=threading.Thread(target=self._loop,name='audit-hash',daemon=True);self._thread.start()

    def submit(self,event,light,frame,counts):
        """Enqueue one tic. The arrays must stay unmodified after submission."""
        if self.failed:raise RuntimeError(self.failed)
        for k in DIGESTS:event.setdefault(k,None)
        handle=AuditHandle(event);item=(handle,light,frame,counts)
        try:self.queue.put_nowait(item)
        except queue.Full:
            self.blocked+=1;self.queue.put(item)
        self.submitted+=1
        return handle

    def _loop(self):
        while True:
            item=self.queue.get()
            if item is None:self.queue.task_done();break
            handle,light,frame,counts=item
            try:
                e=handle.event
                e['input_sha256']=hashlib.sha256(light.tobytes()).hexdigest()
                e['source_frame_sha256']=hashlib.sha256(frame.tobytes()).hexdigest()
                e['spike_counts_sha256']=hashlib.sha256(counts.tobytes()).hexdigest()
                self.logger.info(json.dumps(e,separators=(',',':')))
                self.written+=1
            except Exception as error:handle.error=self.failed=f'{type(error).__name__}: {error}'
            finally:handle.done.set();self.queue.task_done()

    def flush(self):
        """Block until every submitted line has been written."""
        self.queue.join()
        if self.failed:raise RuntimeError(self.failed)

    def close(self,timeout=None):
        try:self.flush()
        finally:
            self.queue.put(None);self._thread.join(timeout=timeout)

    def stats(self):
        return {'submitted':self.submitted,'written':self.written,'blocked':self.blocked,'depth':self.queue.qsize(),
                'maxsize':self.queue.maxsize,'failed':self.failed}
