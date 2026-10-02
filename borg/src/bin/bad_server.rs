mod server_common;

fn main() {
    server_common::run(server_common::ResponseFault::corrupt_reads_from_env());
}
