"""Observe real SLLM backend behavior. Requires fresh external GPU approval."""
import asyncio,hashlib,json,os,queue,signal,socket,subprocess,sys,threading,time,traceback,uuid,faulthandler
from pathlib import Path
Q35=Path('/home/LAB/research-user/engine-route-evaluation-20261016/qwen35-stage-1792137133854569025')
PREVIOUS=Q35.parent/'gpu-stage-1792134963172195358'
CONTROL=Path('/home/LAB/research-user/serverlessllm-work/campaign-20261010-01')
GPU='GPU-00000000-0000-0000-0000-000000000000'
NAME='Qwen3.5-9B-route-research'

def admit(root,engine):
    approval=json.loads((root/'GPU-APPROVAL.json').read_text())
    if approval.get('state')!='active':raise PermissionError('fresh GPU approval is required')
    route=approval['routes'][engine]
    assert time.time()<route['deadline_epoch']<=route['started_epoch']+1200
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
    assert os.environ.get('CUDA_MPS_PIPE_DIRECTORY')=='', 'explicit standalone mode is required'
    compute_mode=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=compute_mode','--format=csv,noheader'],text=True).strip()
    assert compute_mode=='Default', 'standalone multi-process route requires existing default compute mode'
    assert approval['gpu_indices']==[0] and approval['gpu_uuids']==[GPU]
    assert approval['memory_max_bytes']==64*1024**3 and approval['cpu_affinity']==[0,1,2,3]
    groups=Path('/proc/self/cgroup').read_text();assert route['unit']+'.service' in groups
    cg=Path('/sys/fs/cgroup')/groups.strip().split('::')[-1].lstrip('/')
    assert int((cg/'memory.max').read_text())<=64*1024**3
    assert set(os.sched_getaffinity(0))<={0,1,2,3}
    return approval,route,cg

def main():
    engine,root_text=sys.argv[1:];root=Path(root_text)
    approval,route,cg=admit(root,engine) # No engine imports before this guard.
    faulthandler.dump_traceback_later(45, repeat=True)
    output=root/('gpu-'+engine+'-'+str(time.time_ns()));output.mkdir(mode=0o700)
    report={'engine':engine,'started_epoch':time.time(),'scope':'real backend object/worker/Store lifecycle; Controller/Ray/HTTP are not started','operations':[],'status':'started','copy_delta':json.loads((root/'COPY-DELTA.json').read_text())}
    server=None;handle=None;backend=None;samples=[];stop=threading.Event();captures=[]
    def gpu_apps():
        return subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
    def save(): (output/'result.json').write_text(json.dumps(report,indent=2,default=str))
    def sample():
        while not stop.wait(1):
            p=subprocess.run(['nvidia-smi','-i',GPU,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
            samples.append({'epoch':time.time(),'value':p.stdout.strip(),'memory_current':(cg/'memory.current').read_text().strip()})
    def timeout(*args):raise TimeoutError('approved backend route deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(route['deadline_epoch']-time.time()-25)))
    sys.path.insert(0,str(root/'candidate'))
    native=json.loads((Q35/'vllm-native-export-attempt1/result.json').read_text())
    input_ids=native['responses'][0]['input_ids']
    source=Q35/'source/Qwen3.5-9B';tok=json.loads((source/'tokenizer_config.json').read_text());cfg=json.loads((source/'config.json').read_text())
    eos=cfg['text_config']['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
    token=tok['pad_token'];token=token.get('content') if isinstance(token,dict) else token
    pads=[int(k) for k,v in tok['added_tokens_decoder'].items() if v['content']==token];assert len(pads)==1;pad=pads[0]
    report['input_ids']=input_ids;report['eos_token_id']=eos;report['pad_token_id']=pad
    def make_backend(number):
        if engine=='vllm':
            from sllm.backends import VllmBackend
            import sllm.backends.vllm_backend as mod
            if number==1:
                original=mod.process_output
                def observed(value,model_name):
                    captures.append({'request_id':value.request_id,'token_ids':list(value.outputs[0].token_ids),'finish_reason':value.outputs[0].finish_reason,'observed_epoch':time.time()})
                    return original(value,model_name)
                mod.process_output=observed
            return VllmBackend(NAME,{'load_format':'serverless_llm_spike','pretrained_model_name_or_path':str(Q35/'store/vllm/Qwen3.5-9B-attempt01'),'dtype':'bfloat16','enforce_eager':True,'max_model_len':512,'max_num_seqs':1,'max_num_batched_tokens':512,'gpu_memory_utilization':0.85,'enable_prefix_caching':True})
        from sllm.backends import SglangBackend
        from sllm_sglang.config import NO_PLUGINS
        short=Path('/home/LAB/research-user')/('.be-'+uuid.uuid4().hex[:12]);short.mkdir(mode=0o700)
        report.setdefault('private_socket_directories',[]).append(str(short))
        store_index=Q35/'store/transformers/Qwen/Qwen3.5-9B/tensor_index.json'
        env={k:os.environ[k] for k in ['HF_HOME','FLASHINFER_WORKSPACE_BASE','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'] if k in os.environ}
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_MPS_PIPE_DIRECTORY='')
        versions={'sglang':'0.5.21','torch':'2.13.0','transformers':'5.12.1'}
        sources=[p for p in (root/'candidate/sllm_sglang').glob('*.py')]
        config={'model_name':NAME,'model_path':str(source),'storage_path':str(Q35/'store'),'store_model_path':'transformers/Qwen/Qwen3.5-9B','tensor_index_sha256':hashlib.sha256(store_index.read_bytes()).hexdigest(),'allowed_tensor_dtypes':sorted({v[4] for v in json.loads(store_index.read_text()).values()}),'parent_path':str(root/'GPU-APPROVAL.json'),'parent_id':approval['interval_id'],'unit':route['unit'],'phase_deadline_epoch':route['deadline_epoch'],'gpu_index':0,'gpu_uuid':GPU,'runtime_mode':'standalone','runtime_environment':env,'runtime_python':str(PREVIOUS/'sglang-venv/bin/python'),'runtime_pythonpath':[str(root/'candidate'),str(PREVIOUS/'modern-store-client')],'runtime_log':str(output/('worker'+str(number)+'.log')),'runtime_versions':versions,'runtime_file_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'socket_path':str(short/'worker.sock'),'maximum_output_bytes':16*1024**2,'store_address':'127.0.0.1:18573','startup_timeout_seconds':360,'request_timeout_seconds':60,'eos_token_id':eos,'pad_token_id':pad,'prefix_cache':True,'kv_tokens':512,'max_running_requests':1,'attention_backend':'triton'}
        return SglangBackend(NAME,config)
    async def request(b,identity,budget):
        channel=queue.Queue(maxsize=256)
        if engine=='vllm':data={'model':NAME,'request_id':identity,'input_tokens':input_ids,'max_tokens':budget,'temperature':0.0,'ignore_eos':False,'stop_token_ids':eos}
        else:data={'baseline_protocol':'crosspool-preflight-v1','model':NAME,'request_id':identity,'input_ids':input_ids,'max_tokens':budget,'do_sample':False,'eos_token_id':eos,'pad_token_id':pad,'stream':False,'_stream_queue':channel}
        task=asyncio.create_task(b.generate(data))
        return task,channel
    async def active(b,task,channel):
        end=min(time.time()+15,route['deadline_epoch']-30)
        while time.time()<end and not task.done():
            if engine=='vllm':
                tokens=await b.get_current_tokens()
                if tokens:return {'observed':True,'tokens':tokens}
            elif not channel.empty():return {'observed':True,'first_chunk':channel.get_nowait()}
            await asyncio.sleep(0.005)
        return {'observed':False,'request_finished':task.done()}
    async def exercise():
        nonlocal backend
        backend=make_backend(1);t=time.time();await backend.init_backend();report['initial_ready_seconds']=time.time()-t;save()
        tsk,ch=await request(backend,'normal',8);response=await asyncio.wait_for(tsk,60)
        normal={'op':'normal_request','response':response,'stream_chunks':[]}
        while not ch.empty():normal['stream_chunks'].append(ch.get_nowait())
        normal['captured_output']=next((x for x in captures if x['request_id']=='normal'),None)
        actual=(normal['captured_output'] or {}).get('token_ids') if engine=='vllm' else response.get('baseline',{}).get('output_ids')
        normal['expected_token_ids']=native['responses'][0]['output'][0]['token_ids']
        normal['tokens_equal_reference']=actual==normal['expected_token_ids']
        report['operations'].append(normal);save()
        tsk,ch=await request(backend,'trace-and-drain',64);observed=await active(backend,tsk,ch)
        event={'op':'token_trace_and_stop_drain','inflight_observation':observed}
        try:event['tokens']=await backend.get_current_tokens()
        except NotImplementedError:event['token_trace']='not_implemented'
        t=time.time();await backend.stop();event['stop_seconds']=time.time()-t;event['generation_finished_on_stop_return']=tsk.done();event['gpu_apps_after_stop']=gpu_apps()
        try:event['response']=await asyncio.wait_for(tsk,20)
        except BaseException as e:event['generation_error']=type(e).__name__+': '+str(e)
        event['captured_output']=next((x for x in captures if x['request_id']=='trace-and-drain'),None)
        event['drained_normal_response']='error' not in event.get('response',{'error':'no response'})
        report['operations'].append(event);save()
        backend=make_backend(2);await backend.init_backend()
        event={'op':'resume_token_replay'}
        try:await backend.resume_kv_cache([input_ids]);event['result']='returned'
        except NotImplementedError:event['result']='not_implemented'
        report['operations'].append(event);save()
        tsk,ch=await request(backend,'shutdown-active',64);observed=await active(backend,tsk,ch)
        event={'op':'shutdown_during_request','inflight_observation':observed};t=time.time();await backend.shutdown();event['shutdown_seconds']=time.time()-t;event['gpu_apps_after_shutdown']=gpu_apps()
        try:event['response']=await asyncio.wait_for(tsk,20)
        except BaseException as e:event['generation_error']=type(e).__name__+': '+str(e)
        event['captured_output']=next((x for x in captures if x['request_id']=='shutdown-active'),None)
        report['operations'].append(event);report['captures']=captures;save()
    try:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert GPU not in apps,'GPU0 occupied before backend launch'
        threading.Thread(target=sample,daemon=True).start()
        with socket.socket() as sock:sock.bind(('127.0.0.1',18573))
        handle=(output/'store-server.log').open('w')
        env=dict(os.environ,PYTHONPATH=str(CONTROL/'store-buffer-candidate01'),STORAGE_PATH=str(Q35/'store'),SLLM_STORE_ADDRESS='127.0.0.1:18573')
        server=subprocess.Popen([str(CONTROL/'venv/bin/python'),'-c','from sllm_store.cli import main;main()','start','--host','127.0.0.1','--port','18573','--storage-path',str(Q35/'store'),'--mem-pool-size','24GB','--num-thread','2','--chunk-size','32MB'],stdout=handle,stderr=subprocess.STDOUT,env=env)
        import grpc
        channel=grpc.insecure_channel('127.0.0.1:18573');grpc.channel_ready_future(channel).result(timeout=45);channel.close()
        asyncio.run(exercise());report['status']='backend_observation_returned'
    except BaseException:report['status']='failed';report['error']=traceback.format_exc();save()
    finally:
        if backend is not None:
            try:asyncio.run(backend.shutdown())
            except BaseException:report['final_shutdown_error']=traceback.format_exc()
        if server is not None:
            server.terminate()
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
            report['store_exit_code']=server.returncode
        if handle is not None:handle.close()
        stop.set();report['ended_epoch']=time.time();(output/'gpu-samples.json').write_text(json.dumps(samples));save()
    print(json.dumps({'engine':engine,'status':report['status'],'result':str(output/'result.json'),'error_tail':report.get('error','')[-2000:]}),flush=True)
if __name__=='__main__':main()
