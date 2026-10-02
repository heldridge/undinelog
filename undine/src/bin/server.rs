mod server_common;

fn main() {
    server_common::run(server_common::ResponseFault::Honest);
}
