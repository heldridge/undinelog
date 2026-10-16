use paper_a_optimized::{AuthParams,CodedArrayConfig,Fp,build_coded_params,build_shard_registry_deterministic,decode_value,encode_value};
use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::private::{Grouping,PrivateLayout,PirShape,PirClient,PirDatabase,PackKeys,entry_from_cells};
use paper_a_optimized::net::{self,wire::{AuthPrivateReadRequest,AuthPrivateReadResponse,AuthPrivateReadReply,NetByteBlob,encode_binary,decode_binary}};
use paper_a_optimized::private::wire::{pack_words,unpack_words,response_from_net};
use rand::{RngCore,SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde_json::json;
use std::{hint::black_box,time::Instant};
fn request(layout:&PrivateLayout,leaf:u64)->(PirClient,Vec<u8>){
    let client=PirClient::generate(PirShape::of(layout));
    let query=client.query(layout.pattern_of(leaf));
    let keys=client.pack_keys();
    let message=net::Message::AuthReadPrivate{req:AuthPrivateReadRequest{
        leaves:layout.leaves(),kappa:64,pattern_bits:u64::from(layout.group_bits()),
        query:NetByteBlob(pack_words(&query)),pack_keys:NetByteBlob(pack_words(&keys.words))}};
    (client,encode_binary(&message).unwrap())
}
fn parse(request:&[u8])->(Vec<u64>,PackKeys){
    let net::Message::AuthReadPrivate{req}=decode_binary::<net::Message>(request).unwrap() else {panic!()};
    (unpack_words(&req.query.0).unwrap(),PackKeys{words:unpack_words(&req.pack_keys.0).unwrap()})
}
fn serve(db:&PirDatabase,request:&[u8])->Vec<u8>{
    let (query,keys)=parse(request);
    let ciphertexts=db.answer(&query,&keys).unwrap();
    let response=net::Response::AuthReadPrivate{resp:AuthPrivateReadResponse{replies:vec![AuthPrivateReadReply{
        registry_index:0,ciphertexts:ciphertexts.into_iter().map(NetByteBlob).collect()}]}};
    encode_binary(&response).unwrap()
}
fn verify(client:&PirClient,layout:&PrivateLayout,entries:&[Fp],leaf:u64,reply:&[u8]){
    let net::Response::AuthReadPrivate{resp}=decode_binary::<net::Response>(reply).unwrap() else {panic!()};
    let response=response_from_net(resp);
    let slots=client.decode(&response.replies[0].response).unwrap();
    for item in layout.opening_plan(leaf){
        assert_eq!(entry_from_cells(&slots[item.slot],item.entry_in_record),entries[layout.entry_index(item.array_index)]);
    }
}
fn main(){
    let params=build_coded_params(&CodedArrayConfig{kappa:64,shard_count:64,
        shard_registry:build_shard_registry_deterministic(64),party_registry_indices:vec![],seed:42});
    let auth=AuthParams::new(50);
    let leaves=1u64<<27;
    let layout=PrivateLayout::with_grouping(&params,&auth,NodeLinearization::Preorder,leaves,
        Grouping{segments_per_group:1,pattern_bits:14}).unwrap();
    let shape=PirShape::of(&layout);
    let mut rng=ChaCha20Rng::seed_from_u64(42);
    let mut hash=blake3::Hasher::new();
    let entries=(0..layout.records()*layout.entries_per_record()).map(|_|{
        let mut bytes=[0;32];rng.fill_bytes(&mut bytes);let value=decode_value(bytes);hash.update(&encode_value(value));value
    }).collect::<Vec<_>>();
    let input_hash=hash.finalize().to_hex().to_string();
    let cells=PirDatabase::cells_from_entries(&layout,&entries);
    let (db,_)=PirDatabase::from_cells_timed(&layout,cells,1);
    eprintln!("setup complete: {} entries, {} bytes, {} outputs",entries.len(),entries.len()*32,shape.outputs());
    let cases=[("answer",0usize),("answer",1),("answer",2),("first_pass",0),("first_pass",1)];
    let mut samples=Vec::new();
    let mut request_bytes=0;
    let mut reply_bytes=0;
    for trial in 0..23usize{
        for order in 0..cases.len(){
            let case=(order+trial)%cases.len();
            let (operation,scans)=cases[case];
            let leaf=((trial*cases.len()+case) as u64*104729+65537)%leaves;
            let (client,wire)=request(&layout,leaf);
            let(query,_)=parse(&wire);
            for _ in 0..scans{black_box(db.first_pass(&query).unwrap());}
            let ns;
            let reply;
            if operation=="answer"{
                drop(query);
                let start=Instant::now();
                reply=black_box(serve(&db,&wire));
                ns=start.elapsed().as_nanos() as u64;
            }else{
                let start=Instant::now();
                let answer=black_box(db.first_pass(&query).unwrap());
                ns=start.elapsed().as_nanos() as u64;
                assert_eq!(answer.len(),shape.db_cols());
                drop(query);
                reply=serve(&db,&wire);
            }
            verify(&client,&layout,&entries,leaf,&reply);
            request_bytes=wire.len();reply_bytes=reply.len();
            samples.push(json!({"trial":trial,"retained":trial>=3,"operation":operation,"warm_scans":scans,"leaf":leaf,"ns":ns}));
        }
    }
    println!("{}",json!({"description":"Controlled PIR-only diagnostic with identical seeded random primary entries; every decoded path entry checked. No Merkle hashing or database setup is timed.","input_hash":input_hash,"leaves":leaves,"kappa":64,"entry_count":entries.len(),"patterns":shape.patterns,"db_rows":shape.db_rows(),"db_cols":shape.db_cols(),"outputs":shape.outputs(),"request_bytes":request_bytes,"reply_bytes":reply_bytes,"samples":samples}));
}
