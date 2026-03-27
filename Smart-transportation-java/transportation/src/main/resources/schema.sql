
create table if not exists `sensor_check`(
    name varchar(20),
    president varchar(20),
    id int primary key ,
    type varchar(20) default '卡口摄像',
    state int,
    level tinyint,
    address varchar(40),
    update_time timestamp default  current_timestamp on update current_timestamp,
    create_time timestamp default  current_timestamp
);

create table if not exists `sensor_park`(
    name varchar(20),
    president varchar(20),
     id int primary key ,
     type varchar(20) default '违规停车',
     state int,
     level tinyint,
     address varchar(40),
     update_time timestamp default  current_timestamp on update current_timestamp,
     create_time timestamp default  current_timestamp
);

create table if not exists `sensor_speed`(
     name varchar(20),
     president varchar(20),
     id int primary key ,
     type varchar(20) default '区间测速',
     state int,
     level tinyint,
     address varchar(40),
     update_time timestamp default  current_timestamp on update current_timestamp,
     create_time timestamp default  current_timestamp
);

create table if not exists `sensor_flow`(
    name varchar(20),
    president varchar(20),
     id int primary key ,
     type varchar(20) default '流量监测',
     state int,
     level tinyint,
     address varchar(40),
     update_time timestamp default  current_timestamp on update current_timestamp,
     create_time timestamp default  current_timestamp
);


create table if not exists `sensor_environment`(
        name varchar(20),
        president varchar(20),
        id int primary key ,
        type varchar(20) default '环境监测',
        state int,
        level tinyint,
        address varchar(40),
        update_time timestamp default  current_timestamp on update current_timestamp,
        create_time timestamp default  current_timestamp
);
create table if not exists `sensor_lane`(
       name varchar(20),
       president varchar(20),
       id int primary key ,
       type varchar(20) default '车道监测',
       state int,
       level tinyint,
       address varchar(40),
       update_time timestamp default  current_timestamp on update current_timestamp,
       create_time timestamp default  current_timestamp
);

create table if not exists `event`(
    id int auto_increment primary key ,
    type varchar(20),
    state tinyint,
    alarm_type varchar(10),
    level tinyint,
    address varchar(50),
    operation varchar(10),
    content varchar(200),
    source varchar(50),
    reportPerson varchar(20),
    handlePerson varchar(20),
    create_time timestamp default  current_timestamp,
    update_time timestamp default current_timestamp on update  CURRENT_TIMESTAMP,
    handle_time timestamp
)
